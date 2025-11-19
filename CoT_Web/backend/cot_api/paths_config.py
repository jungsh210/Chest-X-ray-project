from pathlib import Path

DPO_JSONL = "/home/jungsh210/Chest_xray/code/CoT_Web/Medical/backend/outputs_cot/dpo_train.jsonl"
FEEDBACK_JSONL = "/home/jungsh210/Chest_xray/code/CoT_Web/Medical/backend/outputs_cot/feedback.jsonl"

BASE_MODEL = "/home/jungsh210/hf_models/Lingshu-7B"
INITIAL_LORA = "/home/jungsh210/Chest_xray/code/1021/Medical_lora/runs/lingshu7b_lora_no_meta" 

IMAGE_ROOT = "/mnt/jungsh/MIMIC_data/mimic-cxr-jpg/2.1.0/files/"
IMAGE_PATTERN = "/mnt/jungsh/MIMIC_data/mimic-cxr-jpg/2.1.0/files/p1[0-3]/*/s*/*.jpg"
META_CSV   = "/mnt/jungsh/MIMIC_data/mimic-cxr-2.0.0-metadata.csv"
PATIENTS_CSV = "/mnt/jungsh/MIMIC_data/patients.csv"

RUNS_ROOT   = Path("/home/jungsh210/Chest_xray/code/CoT_Web/Medical/backend/runs")
RUNS_ROOT.mkdir(parents=True, exist_ok=True)

EVAL1_ROOT = Path("/home/jungsh210/Chest_xray/code/CoT_Web/Medical/backend/outputs_eval1")
EVAL2_ROOT = Path("/home/jungsh210/Chest_xray/code/CoT_Web/Medical/backend/outputs_eval2")
EVAL1_ROOT.mkdir(parents=True, exist_ok=True)
EVAL2_ROOT.mkdir(parents=True, exist_ok=True)

LOGS_DIR = RUNS_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

PAGE_SIZE_DEFAULT = 10

ALLOWED_VIEWS = ["AP", "PA"]

PRED_CACHE_JSON = "/home/jungsh210/Chest_xray/code/CoT_Web/Medical/backend/outputs_cot/pred_cache.json"
