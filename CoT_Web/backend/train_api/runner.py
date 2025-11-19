import os, sys, shutil, subprocess
from pathlib import Path
from typing import Callable, Optional

from .paths_train_config import (
    BASE_MODEL, META_CSV, IMAGE_PATTERN,
    DPO_JSONL, RUNS_ROOT, EVAL1_ROOT, EVAL2_ROOT, LOGS_DIR,
    INITIAL_LORA
)

PY = sys.executable

def _next_version_num() -> int:
    existing = []
    for p in RUNS_ROOT.glob("trl_lora_v*"):
        try:
            n = int(p.name.split("trl_lora_v")[-1])
            existing.append(n)
        except:
            pass
    return (max(existing) + 1) if existing else 1

def _latest_subdir(root: Path) -> Optional[str]:
    dirs = [d for d in root.glob("*") if d.is_dir()]
    if not dirs: return None
    dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(dirs[0])

def _run_cmd(cmd, log_fh, env=None):
    log_fh.write(f"\n[CMD] {' '.join(cmd)}\n"); log_fh.flush()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    for line in proc.stdout:
        log_fh.write(line); log_fh.flush()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")

def _auto_choose_cuda(env: dict):
    """
    CUDA_VISIBLE_DEVICES가 비어있으면 nvidia-smi로 메모리 사용량 가장 낮은 GPU를 선택.
    실패 시 그대로 두고 진행.
    """
    if env.get("CUDA_VISIBLE_DEVICES"):
        return
    try:
        q = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            text=True
        ).strip().splitlines()
        pairs = []
        for row in q:
            idx, used = row.split(",")
            pairs.append((int(idx.strip()), int(used.strip())))
        pairs.sort(key=lambda x: x[1])  # 메모리 사용량 오름차순
        if pairs:
            env["CUDA_VISIBLE_DEVICES"] = str(pairs[0][0])
    except Exception:
        pass

def train_full(*, epochs=1, lr=2e-5, bsz=1, grad_accum=1, limit=10,
               progress_cb: Callable[[int, str, str], None],
               done_cb: Callable[[bool, Optional[str], Optional[str], str], None]):
    """
    전체 파이프라인:
      1) train_trl_offline.py 로 DPO(LoRA) 학습
      2) main_eval.py 로 1차(LoRA=SFT) / 2차(TRL+LoRA) 생성 및 평가
      3) compare_metrics.py 로 성능 비교
      4) 최종 모델 복사 저장(final_trl_model_v*)
    """
    v = _next_version_num()
    trl_dir   = RUNS_ROOT / f"trl_lora_v{v}"
    final_dir = RUNS_ROOT / f"final_trl_model_v{v}"
    log_path  = LOGS_DIR / f"train_v{v}.log"

    os.makedirs(LOGS_DIR, exist_ok=True)

    # 공통 ENV
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("NCCL_P2P_DISABLE", "1")
    env.setdefault("NCCL_IB_DISABLE", "1")
    _auto_choose_cuda(env)

    try:
        with open(log_path, "w", encoding="utf-8") as log_fh:
            log_fh.write(f"[INFO] Start training v{v}\n")
            log_fh.write(f"[INFO] DPO_JSONL: {DPO_JSONL}\n")
            log_fh.write(f"[INFO] Base LoRA (1st gen): {INITIAL_LORA}\n")
            log_fh.flush()

            progress_cb(5, "dpo_train", "DPO 학습 준비")

            # 1) TRL DPO 학습
            if trl_dir.exists():
                shutil.rmtree(trl_dir)
            progress_cb(10, "dpo_train", "DPO 학습 시작")
            _run_cmd([
                PY, str((Path(__file__).parent / "train_trl_offline.py").resolve()),
                "--base_model", BASE_MODEL,
                "--dpo_jsonl", DPO_JSONL,
                "--out_dir", str(trl_dir),
                "--epochs", str(epochs),
                "--bsz", str(bsz),
                "--grad_accum", str(grad_accum),
                "--lr", str(lr),
                "--bf16"
            ], log_fh, env=env)
            log_fh.write(f"[OK] DPO LoRA saved to: {trl_dir}\n"); log_fh.flush()
            progress_cb(60, "dpo_train", "DPO 학습 완료")

            # 2) 평가
            progress_cb(65, "eval1", "1차 판독문 평가(LoRA-SFT)")
            try:
                _run_cmd([
                    PY, "main_eval.py",
                    "--models", "7B",
                    "--prompt_types", "Existing",
                    "--meta", "meta",
                    "--limit", str(limit),
                    "--lora_path", str(INITIAL_LORA),
                    "--patterns", IMAGE_PATTERN
                ], log_fh, env=env)
            except Exception as e:
                log_fh.write(f"[WARN] eval1 skipped: {e}\n"); log_fh.flush()

            progress_cb(75, "eval2", "2차 판독문 평가(TRL+LoRA)")
            try:
                _run_cmd([
                    PY, "main_eval.py",
                    "--models", "7B",
                    "--prompt_types", "Existing",
                    "--meta", "meta",
                    "--limit", str(limit),
                    "--lora_path", str(trl_dir),
                    "--patterns", IMAGE_PATTERN
                ], log_fh, env=env)
            except Exception as e:
                log_fh.write(f"[WARN] eval2 skipped: {e}\n"); log_fh.flush()

            # 3) 비교
            progress_cb(85, "compare", "성능 비교")
            try:
                first_dir  = _latest_subdir(EVAL1_ROOT)
                second_dir = _latest_subdir(EVAL2_ROOT)
                if first_dir and second_dir:
                    m1 = Path(first_dir) / "metrics.json"
                    m2 = Path(second_dir) / "metrics.json"
                    if m1.exists() and m2.exists():
                        _run_cmd([PY, "compare_metrics.py", str(m1), str(m2)], log_fh, env=env)
                    else:
                        log_fh.write("[WARN] metrics.json not found for comparison.\n")
                else:
                    log_fh.write("[WARN] eval1/eval2 not found.\n")
                log_fh.flush()
            except Exception as e:
                log_fh.write(f"[WARN] compare skipped: {e}\n"); log_fh.flush()

            # 4) 최종 저장
            progress_cb(92, "save", "최종 모델 저장")
            if final_dir.exists():
                shutil.rmtree(final_dir)
            shutil.copytree(trl_dir, final_dir)
            log_fh.write(f"[OK] Saved final model: {final_dir}\n"); log_fh.flush()
            progress_cb(100, "done", "완료")

        done_cb(True, str(final_dir), str(log_path), f"✅ 모델 저장 완료 (경로: {final_dir})")
        # 선택적으로 결과 dict 리턴도 가능하지만, app_train에서 콜백을 통해 이미 처리함.
        return {
            "version": v,
            "trl_dir": str(trl_dir),
            "final_model_dir": str(final_dir),
            "log": str(log_path),
        }
    except Exception as e:
        try:
            with open(log_path, "a", encoding="utf-8") as log_fh:
                log_fh.write(f"\n[ERROR] {e}\n")
        except:
            pass
        done_cb(False, None, str(log_path), f"❌ 학습 실패: {e}")
        # 예외는 상위에서 메시지로 처리하므로 여기서는 추가 전파하지 않음.
        return None