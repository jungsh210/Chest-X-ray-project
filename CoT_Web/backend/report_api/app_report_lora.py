# -*- coding: utf-8 -*-
# report_api/app_report_lora.py  (9001: LoRA 서버)
import os, uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Query, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .service_core import (
    full_pipeline, generate_report, get_demographics, _safe_join, run_meditron
)
from .paths_report_config import DEFAULT_LORA_PATH

app = FastAPI(title="Report API - LoRA", version="1.0")

# ---- CORS ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ---- Upload dir ----
UPLOAD_DIR = Path("/home/jungsh210/Chest_xray/code/CoT_Web/Medical/backend/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
_ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


# ---- Schemas ----
class GenerateReq(BaseModel):
    rel: Optional[str] = None
    image_path: Optional[str] = None
    prompt_type: str = "Existing"     # "Existing" | "Strong"
    use_meta: bool = True
    original_report: str = ""         # (옵션) 원문


# ---- Health ----
@app.get("/health")
def health():
    return {"ok": True, "mode": "lora", "default_lora": DEFAULT_LORA_PATH}


# ---- Upload ----
@app.post("/api/report/upload")
async def api_upload(file: UploadFile = File(...)):
    # 확장자 결정
    orig = file.filename or "image.jpg"
    ext = os.path.splitext(orig)[1].lower()
    if ext not in _ALLOWED_EXT:
        ext = ".jpg"
    # 파일 저장
    fname = f"{uuid.uuid4().hex}{ext}"
    dst = UPLOAD_DIR / fname
    try:
        with open(dst, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    except Exception as e:
        raise HTTPException(500, f"Upload failed: {e}")

    return {"ok": True, "path": str(dst)}


# ---- Demographic (CSV/모델 기반) ----
@app.get("/api/report/demographic")
def api_demographic(rel: Optional[str] = None, image_path: Optional[str] = None):
    if not rel and not image_path:
        raise HTTPException(status_code=400, detail="Provide 'rel' or 'image_path'.")
    img = _safe_join(image_path or rel)
    demo = get_demographics(str(img))
    return {
        "ok": True,
        "demographics": {
            "age_years": demo.age_years,
            "sex": demo.sex,
            "view": demo.view,
        },
        "image_path": str(img),
    }


# ---- LoRA 1차 보고서 생성 (이미지 경로 필요) ----
@app.post("/api/report/generate")
def api_generate(req: GenerateReq):
    if not (req.image_path or req.rel):
        raise HTTPException(400, "image_path or rel required.")

    data = full_pipeline(
        rel=req.rel,
        image_path=req.image_path,
        use_trl_latest=False,          # LoRA 전용
        fixed_lora=DEFAULT_LORA_PATH,  # 기본 LoRA
        prompt_type=req.prompt_type,
        use_meta=req.use_meta,
        include_meditron=False,
        original_report=req.original_report or "",
    )
    # 프론트에서 기대하는 키 이름을 명확히 반환
    return {
        "ok": True,
        "image_path": data.get("image_path"),
        "report_lora": data.get("report_lora"),
        "demographics": data.get("demographics", {}),
    }


# ---- LoRA + Meditron까지 한 번에 ----
@app.post("/api/report/full")
def api_full(req: GenerateReq):
    if not (req.image_path or req.rel):
        raise HTTPException(400, "image_path or rel required.")

    data = full_pipeline(
        rel=req.rel,
        image_path=req.image_path,
        use_trl_latest=False,          # LoRA 전용
        fixed_lora=DEFAULT_LORA_PATH,
        prompt_type=req.prompt_type,
        use_meta=req.use_meta,
        include_meditron=True,         # 여기서는 감별/추가검사 포함
        original_report=req.original_report or "",
    )
    return {"ok": True, **data}


# ---- (옵션) LoRA 보고서 단독 + 즉시 Meditron 호출 버전 ----
@app.post("/api/report/meditron")
def api_meditron(req: GenerateReq, lingshu_report: str = Query(..., description="Findings/Impression 텍스트")):
    img = _safe_join(req.image_path or req.rel) if (req.image_path or req.rel) else None
    demo = get_demographics(str(img)) if img else None
    med = run_meditron(lingshu_report, original_report=req.original_report or "", demographics=demo)
    return {"ok": True, "meditron": med, "image_path": (str(img) if img else None)}
