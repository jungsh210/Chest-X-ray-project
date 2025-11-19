import os, re, glob
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from .service_core_trl import full_pipeline, generate_report, get_demographics, run_meditron, _safe_join

from pathlib import Path
from datetime import datetime
from pydantic import BaseModel
import json

RUNS_ROOT = "/home/jungsh210/Chest_xray/code/CoT_Web/Medical/backend/runs"

def _latest_final_trl(root: str = RUNS_ROOT) -> Optional[str]:
    """가장 최신 버전의 final_trl_model_v* 폴더 자동 선택"""
    cands = [p for p in glob.glob(os.path.join(root, "final_trl_model_v*")) if os.path.isdir(p)]
    if not cands:
        return None

    def _ver(p: str) -> int:
        m = re.search(r"final_trl_model_v(\d+)$", p)
        return int(m.group(1)) if m else -1

    cands.sort(key=lambda p: (_ver(p), os.path.getmtime(p)), reverse=True)
    return cands[0]

DEFAULT_TRL_LORA_PATH = (
    os.environ.get("REPORT_TRL_LORA")
    or _latest_final_trl()
    or "/home/jungsh210/Chest_xray/code/CoT_Web/Medical/backend/runs/final_trl_model_v1"
)

app = FastAPI(title="Report API - TRL+LoRA", version="1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
FEEDBACK_PATH = BASE_DIR / "data" / "feedback.jsonl"

class GenerateReq(BaseModel):
    rel: Optional[str] = None
    image_path: Optional[str] = None
    prompt_type: str = "Existing"
    use_meta: bool = True
    original_report: str = ""

class FeedbackPayload(BaseModel):
  image_name: str | None = None
  better: str  # "lora" or "trl"
  reward: float
  score_clinic: float | None = None
  score_natural: float | None = None
  score_logic: float | None = None
  report_lora: str | None = None
  report_trl: str | None = None
  meta: dict | None = None

@app.get("/health")
def health():
    latest = _latest_final_trl()
    return {
        "ok": True,
        "mode": "trl_lora",
        "default_trl_lora": latest or DEFAULT_TRL_LORA_PATH,
    }

@app.post("/api/report/full")
def api_full(req: GenerateReq, use_latest: bool = Query(True)):
    """9002 서버 메인 엔드포인트"""
    from .service_core_trl import full_pipeline
    data = full_pipeline(
        rel=req.rel, image_path=req.image_path,
        use_trl_latest=use_latest,
        fixed_lora=None,
        prompt_type=req.prompt_type,
        use_meta=req.use_meta,
        include_meditron=True,
        original_report=req.original_report or "",
    )
    return {"ok": True, **data}

@app.post("/api/report/meditron")
def api_meditron(req: GenerateReq, lingshu_report: str = Query(..., description="Findings/Impression 텍스트")):
    img = _safe_join(req.image_path or req.rel) if (req.image_path or req.rel) else None
    demo = get_demographics(str(img)) if img else None
    med = run_meditron(lingshu_report, original_report=req.original_report or "", demographics=demo)
    return {"ok": True, "meditron": med, "image_path": (str(img) if img else None)}

@app.post("/api/report/feedback")
def save_feedback(item: FeedbackPayload):
    """
    웹 UI에서 입력한 10점 척도 + LoRA/TRL 선호 + 판독문 텍스트를
    feedback.jsonl에 1줄씩 저장.
    DPO/TRL 학습 시 바로 사용할 수 있도록 chosen/rejected도 함께 생성.
    """
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 어떤 쪽이 선택됐는지에 따라 chosen / rejected 설정
    if item.better == "trl":
        chosen = item.report_trl or ""
        rejected = item.report_lora or ""
    else:
        chosen = item.report_lora or ""
        rejected = item.report_trl or ""

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "image_name": item.image_name,
        "better": item.better,          # "lora" or "trl"
        "reward": item.reward,          # 평균 점수
        "scores": {
            "clinic": item.score_clinic,
            "natural": item.score_natural,
            "logic": item.score_logic,
        },
        "reports": {
            "lora": item.report_lora,
            "trl": item.report_trl,
        },
        # DPO/TRL 학습용 쌍
        "dpo": {
            "chosen": chosen,   # 의료진이 더 좋다고 선택한 판독문
            "rejected": rejected,  # 덜 좋다고 판단된 판독문 (없으면 빈 문자열)
        },
        # 환자 메타 정보도 같이 보존
        "meta": item.meta or {},
    }

    # 한 줄 JSONL로 append
    with FEEDBACK_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {"ok": True}
