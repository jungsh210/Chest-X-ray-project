# -*- coding: utf-8 -*-
import os, threading, time
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .paths_train_config import (
    DPO_JSONL, FEEDBACK_JSONL, RUNS_ROOT, LOGS_DIR,
    BASE_MODEL, INITIAL_LORA
)
from . import runner as train_runner  # runner.train_full 사용

app = FastAPI(title="TRAIN API", version="2.1")

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://0.0.0.0:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 전역 상태 ---
_status = {
    "running": False,
    "stage": "idle",            # idle | dpo_train | eval1 | eval2 | compare | save | done | error
    "message": "idle",
    "progress": 0,              # 0~100
    "output_dir": None,
    "log_path": None,
    "version": 0,
}
_last_result = None
_lock = threading.Lock()

def _file_nonempty(p: str | Path) -> bool:
    try:
        return Path(p).is_file() and Path(p).stat().st_size > 0
    except Exception:
        return False

def _count_lines(p: str | Path) -> int:
    try:
        with open(p, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0

def _update(**kw):
    with _lock:
        _status.update(kw)

def _log_tail(path: Optional[str], n: int) -> List[str]:
    if not path or not Path(path).is_file():
        return []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        return [ln.rstrip("\n") for ln in lines[-n:]]
    except Exception:
        return []

def _gpu_env():
    """
    GPU 자동 사용을 위한 기본 환경 변수.
    실제 여유 GPU 선택은 runner 쪽에서 수행(필요 시).
    """
    env = os.environ.copy()
    # 프리셋이 없으면 다 보이도록 두고, PyTorch 메모리 설정만 추가
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("NCCL_P2P_DISABLE", "1")
    env.setdefault("NCCL_IB_DISABLE", "1")
    return env

class TrainReq(BaseModel):
    epochs: int = 1
    lr: float = 2e-5
    bsz: int = 1
    grad_accum: int = 1
    limit: int = 10

@app.get("/api/train/can_train")
def can_train():
    """
    프론트가 버튼 노출 여부 판단에 사용.
    dpo_train.jsonl / feedback.jsonl 둘 중 하나라도 내용이 있으면 OK.
    """
    ready = _file_nonempty(DPO_JSONL) or _file_nonempty(FEEDBACK_JSONL)
    return {
        "ok": True,
        "ready": bool(ready),
        "dpo_lines": _count_lines(DPO_JSONL),
        "fb_lines": _count_lines(FEEDBACK_JSONL),
        "base_model": BASE_MODEL,
        "initial_lora": INITIAL_LORA,
    }

@app.get("/api/train/status")
def get_status():
    with _lock:
        return dict(_status)

@app.get("/api/train/log_tail")
def get_log_tail(n: int = Query(200, ge=20, le=2000)):
    with _lock:
        lp = _status.get("log_path")
    return {"ok": True, "lines": _log_tail(lp, n)}

@app.get("/api/train/results")
def get_results():
    return {"ok": True, "result": _last_result}

def _progress_watcher(log_path: str):
    """
    로그를 훑어보며 대략적인 진행률/단계를 갱신.
    runner.py에서 남기는 키워드 기반의 간단한 휴리스틱.
    """
    while True:
        time.sleep(2.0)
        with _lock:
            running = _status["running"]
        if not running:
            break

        lines = _log_tail(log_path, 2000)
        text = "\n".join(lines)

        prog = 5
        stage = "dpo_train"
        msg = "모델 학습 중..."

        if " [CMD] " in text and "train_trl_offline.py" in text:
            prog = max(prog, 10)
        if "[OK] DPO LoRA saved to:" in text:
            prog = max(prog, 60)
            stage = "eval1"
            msg = "1차 생성/평가 중..."

        if "main_eval.py" in text and "--lora_path" not in text:
            prog = max(prog, 65)
        if "main_eval.py" in text and "--lora_path" in text:
            prog = max(prog, 75)
            stage = "eval2"
            msg = "2차 생성/평가 중..."

        if "compare_metrics.py" in text:
            prog = max(prog, 85)
            stage = "compare"
            msg = "성능 비교 중..."

        if "[OK] Saved final model:" in text:
            prog = 100
            stage = "done"
            msg = "완료"

        _update(progress=prog, stage=stage, message=msg)

@app.post("/api/train/start_full")
def start_full(req: TrainReq):
    with _lock:
        if _status["running"]:
            return {"ok": False, "message": "이미 학습이 진행 중입니다."}
        if not (_file_nonempty(DPO_JSONL) or _file_nonempty(FEEDBACK_JSONL)):
            return {"ok": False, "message": "학습할 피드백/DPO 샘플이 없습니다."}

        _status.update({
            "running": True,
            "stage": "dpo_train",
            "message": "모델 학습 중...",
            "progress": 0,
            "output_dir": None,
            "log_path": None,
        })

    # 콜백 정의: runner.train_full → 여기 상태 갱신
    def progress_cb(progress: int, stage: str, message: str):
        _update(progress=progress, stage=stage, message=message or "")

    def done_cb(success: bool, final_dir: Optional[str], log_path: Optional[str], message: str):
        if success:
            _update(stage="done", message=message or "완료", progress=100, output_dir=final_dir, log_path=log_path)
        else:
            _update(stage="error", message=message or "실패", progress=0)
        _update(running=False)

    def _worker():
        global _last_result
        try:
            os.environ.update(_gpu_env())
            res = train_runner.train_full(
                epochs=req.epochs,
                lr=req.lr,
                bsz=req.bsz,
                grad_accum=req.grad_accum,
                limit=req.limit,
                progress_cb=progress_cb,
                done_cb=done_cb,
            )
            _last_result = res
            # log_path가 아직 비어있다면 보정
            logs = sorted(LOGS_DIR.glob("train_v*.log"), key=lambda p: p.stat().st_mtime)
            if logs:
                _update(log_path=str(logs[-1]))
        except Exception as e:
            done_cb(False, None, None, f"❌ 학습 실패: {e}")

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

    # watcher: 로그 파일이 생기면 tail 기반으로 진행률 보강
    def _spawn_watcher():
        for _ in range(15):  # 최대 ~30초 대기(2초 간격)
            time.sleep(2.0)
            logs = sorted(LOGS_DIR.glob("train_v*.log"), key=lambda p: p.stat().st_mtime)
            if logs:
                lp = str(logs[-1])
                _update(log_path=lp)
                threading.Thread(target=_progress_watcher, args=(lp,), daemon=True).start()
                return

    threading.Thread(target=_spawn_watcher, daemon=True).start()
    return {"ok": True, "message": "학습 시작"}
