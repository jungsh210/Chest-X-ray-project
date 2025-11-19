# -*- coding: utf-8 -*-
import os, io, json, glob, re, time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Any

from fastapi import HTTPException
from PIL import Image

import requests 

from .paths_report_config import (
    BASE_MODEL, DEFAULT_LORA_PATH, DEFAULT_TRL_LORA_PATH,
    IMAGE_ROOT, META_CSV, PATIENTS_CSV, ALLOWED_VIEWS, RUNS_ROOT
)

# 로컬 모듈(네가 준 파일을 /infer 폴더로 넣어둠)
from .infer.lingshu_infer import run_lingshu
from .infer.Meditron import Meditron as MeditronRun

# Meditron HTTP 호출 URL
MEDITRON_URL = os.environ.get(
    "MEDITRON_URL", "http://localhost:9102/api/meditron/analyze"
)


_IMAGE_ROOT = Path(IMAGE_ROOT).resolve()

def _safe_join(rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    if not p.is_absolute():
        p = (_IMAGE_ROOT / rel_or_abs).resolve()
    if not str(p).startswith(str(_IMAGE_ROOT)):
        raise HTTPException(status_code=400, detail="Invalid path")
    return p

def _parse_ids_from_rel(rel: str) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    # 예: p10/p10000032/s50414267/xxx.jpg
    parts = rel.split("/")
    sid = stid = did = None
    for i, p in enumerate(parts):
        if p.startswith("p") and len(p) == 9 and p[1:].isdigit():
            try:
                sid = int(p[1:])
            except Exception:
                sid = None
            if i + 1 < len(parts):
                nxt = parts[i + 1]
                if nxt.startswith("s") and nxt[1:].isdigit():
                    try:
                        stid = int(nxt[1:])
                    except Exception:
                        stid = None
    if parts:
        did = Path(parts[-1]).stem
    return sid, stid, did

# ===== CSV 메타 조회 (CoT와 동일 로직) =====
_view_index_cache: Optional[Dict[Tuple[int,int,str], str]] = None
_patient_index_cache: Optional[Dict[int, Dict[str, Optional[str]]]] = None

def _build_view_index() -> Optional[Dict[Tuple[int,int,str], str]]:
    global _view_index_cache
    if _view_index_cache is not None:
        return _view_index_cache
    if not META_CSV or not os.path.exists(META_CSV):
        _view_index_cache = None
        return None
    import pandas as pd
    usecols = ["subject_id","study_id","dicom_id","ViewPosition"]
    df = pd.read_csv(META_CSV, usecols=usecols)
    df["ViewPosition"] = df["ViewPosition"].astype(str).str.upper().str.strip()
    idx = {}
    for _, r in df.iterrows():
        try:
            sid = int(r["subject_id"]); stid = int(r["study_id"]); did = str(r["dicom_id"])
            idx[(sid, stid, did)] = str(r["ViewPosition"])
        except Exception:
            continue
    _view_index_cache = idx
    return idx

def _build_patient_index() -> Optional[Dict[int, Dict[str, Optional[str]]]]:
    global _patient_index_cache
    if _patient_index_cache is not None:
        return _patient_index_cache
    if not PATIENTS_CSV or not os.path.exists(PATIENTS_CSV):
        _patient_index_cache = None
        return None
    import pandas as pd
    usecols = ["subject_id", "anchor_age", "gender"]
    df = pd.read_csv(PATIENTS_CSV, usecols=usecols)

    def _map_sex(x: str) -> Optional[str]:
        if not isinstance(x, str): return None
        x = x.strip().upper()
        if x in ("M","MALE"): return "M"
        if x in ("F","FEMALE"): return "F"
        return None

    idx = {}
    for _, r in df.iterrows():
        try:
            sid = int(r["subject_id"])
        except Exception:
            continue
        age = None
        try:
            age = float(r["anchor_age"])
        except Exception:
            pass
        idx[sid] = {"age_years": (None if age is None else round(age,1)),
                    "sex": _map_sex(str(r["gender"]))}
    _patient_index_cache = idx
    return idx

def lookup_meta_from_csv(rel_or_abs: str) -> Dict[str, Optional[str]]:
    # rel/abs 모두 수용
    p = Path(rel_or_abs)
    if p.is_absolute():
        rel = str(p.resolve()).replace(str(_IMAGE_ROOT)+"/", "")
    else:
        rel = rel_or_abs
    sid, stid, did = _parse_ids_from_rel(rel)
    age = None; sex = None; view = None
    pidx = _build_patient_index()
    if pidx and sid is not None:
        rec = pidx.get(sid, {})
        age = rec.get("age_years", None)
        sex = rec.get("sex", None)
    vidx = _build_view_index()
    if vidx and sid is not None and stid is not None and did is not None:
        vp = vidx.get((sid, stid, did), None)
        if isinstance(vp, str) and vp.upper() in ALLOWED_VIEWS:
            view = vp.upper()
    return {"age_years": age, "sex": sex, "view": view}

# ===== TRL 최신 모델 자동 선택(옵션) =====
def _auto_latest_trl_dir(default_path: str) -> str:
    try:
        cands = []
        # final_trl_model_v* 또는 trl_lora_v* 모두 고려
        for pat in ("final_trl_model_v*", "trl_lora_v*"):
            cands.extend([p for p in RUNS_ROOT.glob(pat) if p.is_dir()])
        if not cands:
            return default_path
        cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return str(cands[0])
    except Exception:
        return default_path

# ===== 파이프라인 =====

@dataclass
class Demographics:
    age_years: Optional[float] = None
    sex: Optional[str] = None
    view: Optional[str] = None

def get_demographics(image_path: str) -> Demographics:
    """CSV 기반 메타(가능 시 AP/PA/age/sex)"""
    meta = lookup_meta_from_csv(image_path)
    age = meta.get("age_years"); sex = meta.get("sex"); view = meta.get("view")
    return Demographics(age_years=age, sex=sex, view=view)

def generate_report(image_path: str,
                    prompt_type: str = "Existing",
                    use_meta: bool = True,
                    lora_path: Optional[str] = None) -> str:
    """Lingshu(+LoRA)로 Findings/Impression 생성"""
    if prompt_type not in ("Existing", "Strong"):
        raise HTTPException(status_code=400, detail="prompt_type must be 'Existing' or 'Strong'")
    report = run_lingshu(
        model_path=BASE_MODEL,
        image_path=image_path,
        prompt_type=prompt_type,
        use_meta=use_meta,
        meta_csv=META_CSV,
        lora_path=lora_path,
    )
    if not report:
        raise HTTPException(status_code=500, detail="Report generation failed (empty result).")
    return report

def run_meditron(lingshu_report: str,
                 original_report: str = "",
                 demographics: Optional[Demographics] = None) -> Dict[str, Any]:
    """Medtiron으로 감별진단/추가검사 JSON 생성"""
    dem = {}
    if demographics:
        if demographics.age_years is not None: dem["age_years"] = demographics.age_years
        if demographics.sex: dem["sex"] = demographics.sex
    json_str = MeditronRun(lingshu_report_text=lingshu_report,
                           original_report_text=original_report,
                           demographics=dem or None)
    try:
        return json.loads(json_str)
    except Exception:
        return {"differential": [], "next_tests": []}

# ===== 요청 단위 end-to-end =====
def full_pipeline(rel: Optional[str] = None,
                  image_path: Optional[str] = None,
                  *,
                  use_trl_latest: bool = False,
                  fixed_lora: Optional[str] = None,
                  prompt_type: str = "Existing",
                  use_meta: bool = True,
                  include_meditron: bool = True,
                  original_report: str = "") -> Dict[str, Any]:
    """
    1) 이미지 경로 확보(rel 또는 절대경로)
    2) demographics(csv)
    3) Lingshu(+LoRA) 생성
    4) Meditron 추천(선택)
    """
    if not rel and not image_path:
        raise HTTPException(status_code=400, detail="Provide 'rel' or 'image_path'.")
    img = _safe_join(image_path or rel)

    # lora 선택
    if fixed_lora is not None:
        lora = fixed_lora
    else:
        lora = DEFAULT_LORA_PATH
    if use_trl_latest:
        lora = _auto_latest_trl_dir(DEFAULT_TRL_LORA_PATH)

    t0 = time.time()
    demo = get_demographics(str(img))
    t_demo = (time.time() - t0) * 1000.0

    t1 = time.time()
    report = generate_report(str(img), prompt_type=prompt_type, use_meta=use_meta, lora_path=lora)
    t_rep = (time.time() - t1) * 1000.0

    med = None
    t_med = 0.0
    if include_meditron:
        t2 = time.time()
        med = run_meditron(report, original_report=original_report, demographics=demo)
        t_med = (time.time() - t2) * 1000.0

    return {
        "image_path": str(img),
        "demographics": {"age_years": demo.age_years, "sex": demo.sex, "view": demo.view},
        "report": report,
        "meditron": med,
        "timing_ms": {"demographics": t_demo, "report": t_rep, "meditron": t_med, "total": (time.time()-t0)*1000.0},
        "lora_used": lora,
        "prompt_type": prompt_type,
        "use_meta": use_meta,
    }

def run_meditron(lingshu_report: str,
                 original_report: str = "",
                 demographics: Optional[Demographics] = None) -> Dict[str, Any]:
    """Meditron API(9102)로 감별진단/추가검사 요청"""
    dem = {}
    if demographics:
        if demographics.age_years is not None:
            dem["age_years"] = demographics.age_years
        if demographics.sex:
            dem["sex"] = demographics.sex
        if demographics.view:
            dem["view"] = demographics.view

    payload = {
        "lingshu_report_text": lingshu_report,
        "original_report_text": original_report or "",
        "demographics": dem or None,
    }

    try:
        r = requests.post(MEDITRON_URL, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        return data.get("result", data)
    except Exception as e:
        print(f"[Meditron] call failed: {e}")
        return {"differential": [], "next_tests": [], "notes": ""}