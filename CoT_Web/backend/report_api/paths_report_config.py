from pathlib import Path

# ===== Base/LoRA 경로 =====
BASE_MODEL = "/home/jungsh210/hf_models/Lingshu-7B"

# LoRA 페이지(1페이지) 기본 LoRA
DEFAULT_LORA_PATH = "/home/jungsh210/Chest_xray/code/1021/Medical_lora/runs/lingshu7b_lora_no_meta"

# TRL+LoRA 페이지(2페이지) 기본 LoRA (최종/최근 경로로 교체 가능)
#   - 학습 파이프라인이 저장하는 최종 디렉토리 중 최신 것을 자동으로 쓰고 싶다면,
#     service_core.py 내 _auto_latest_trl_dir() 사용(아래 앱에서 True 주입)
DEFAULT_TRL_LORA_PATH = "/home/jungsh210/Chest_xray/code/CoT_Web/Medical/backend/runs/final_trl_model_v5"

# ===== 메타/이미지 경로 (CoT와 일치) =====
IMAGE_ROOT   = "/mnt/jungsh/MIMIC_data/mimic-cxr-jpg/2.1.0/files/"
META_CSV     = "/mnt/jungsh/MIMIC_data/mimic-cxr-2.0.0-metadata.csv"
PATIENTS_CSV = "/mnt/jungsh/MIMIC_data/patients.csv"

# ===== 기타 =====
ALLOWED_VIEWS = ["AP", "PA"]

# (옵션) 최종 TRL 모델들이 모여있는 루트 (최신 버전 자동 선택용)
RUNS_ROOT = Path("/home/jungsh210/Chest_xray/code/CoT_Web/Medical/backend/runs")