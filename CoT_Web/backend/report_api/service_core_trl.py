# report_api/service_core_trl.py
import os, json
from dataclasses import dataclass

import requests

from .infer.lingshu_infer_trl import run_lingshu_trl_textonly
from .infer.utils import load_view_from_meta

from .infer.Meditron import Meditron as MeditronRun

MEDITRON_URL = os.environ.get(
    "MEDITRON_URL", "http://localhost:9102/api/meditron/analyze"
)

@dataclass
class Demo:
    age_years: float | None = None
    sex: str | None = None
    view: str | None = None

def _safe_join(path_or_rel: str | None) -> str | None:
    if not path_or_rel:
        return None
    return os.path.abspath(path_or_rel)

def get_demographics(image_path: str | None) -> Demo:
    # 필요하면 여기서 meta_csv 기반 view까지 읽어와도 됨
    return Demo(age_years=None, sex=None, view=None)

def run_meditron(
    refined_report: str,
    original_report: str = "",
    demographics: Demo | None = None,
) -> dict:
    dem = {}
    if demographics:
        if demographics.age_years is not None:
            dem["age_years"] = demographics.age_years
        if demographics.sex:
            dem["sex"] = demographics.sex
        if demographics.view:
            dem["view"] = demographics.view

    payload = {
        "lingshu_report_text": refined_report,
        "original_report_text": original_report or "",
        "demographics": dem or None,
    }

    try:
        r = requests.post(MEDITRON_URL, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        return data.get("result", data)
    except Exception as e:
        print(f"[Meditron-TRL] call failed: {e}")
        return {"differential": [], "next_tests": [], "notes": ""}

def generate_report(
    prev_report: str,
    view: str | None = None,
    age: float | None = None,
    sex: str | None = None,
    style_hint: str | None = None,
    cot_hint: str | None = None,
    trl_base: str = "/home/jungsh210/hf_models/Lingshu-7B",
    trl_lora: str = "/home/jungsh210/Chest_xray/code/CoT_Web/Medical/backend/runs/final_trl_model_v5",
) -> str:
    return run_lingshu_trl_textonly(
        model_path=trl_base,
        prev_report=prev_report,
        view=view,
        age=age,
        sex=sex,
        lora_path=trl_lora,
        style_hint=style_hint,
        cot_hint=cot_hint,
    )

def full_pipeline(
    rel: str | None = None,
    image_path: str | None = None,
    use_trl_latest: bool = True,
    fixed_lora: str | None = None,
    prompt_type: str = "Existing",
    use_meta: bool = True,
    include_meditron: bool = True,
    original_report: str = "",
    meta_csv: str | None = None,
) -> dict:
    if not original_report or original_report.strip() == "":
        raise RuntimeError("original_report(1차 보고서)가 없습니다. 9001에서 생성 후 전달해 주세요.")

    img = _safe_join(image_path or rel)

    view = None
    if use_meta and img and meta_csv:
        try:
            view = load_view_from_meta(img, meta_csv)
        except Exception:
            view = None

    refined = generate_report(
        prev_report=original_report,
        view=view,
        age=None,
        sex=None,
        style_hint=None,
        cot_hint=None,
        trl_base="/home/jungsh210/hf_models/Lingshu-7B",
        trl_lora=(fixed_lora or "/home/jungsh210/Chest_xray/code/CoT_Web/Medical/backend/runs/final_trl_model_v5"),
    )

    out = {
        "image_path": img,
        "report_lora": original_report,
        "report_trl": refined,
    }

    if include_meditron:
        demo = get_demographics(img)
        med = run_meditron(
            refined_report=refined,
            original_report=original_report,
            demographics=demo,
        )
        out["meditron"] = med

    return out
