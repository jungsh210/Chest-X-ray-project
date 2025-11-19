from pathlib import Path

INITIAL_LORA = "/home/jungsh210/Chest_xray/code/1021/Medical_lora/runs/lingshu7b_lora_no_meta"
EXISTING_TRL = "/home/jungsh210/Chest_xray/code/1104-1/runs/trl_lora_v1"

BASE_MODEL = "/home/jungsh210/hf_models/Lingshu-7B"

DATA_DIR = (Path(__file__).resolve().parent.parent / "outputs_cot")
DATA_DIR.mkdir(parents=True, exist_ok=True)
DPO_JSONL = str(DATA_DIR / "dpo_train.jsonl")
FEEDBACK_JSONL = str(DATA_DIR / "feedback.jsonl")

META_CSV   = "/mnt/jungsh/MIMIC_data/mimic-cxr-2.0.0-metadata.csv"
IMAGE_PATTERN = "/mnt/jungsh/MIMIC_data/mimic-cxr-jpg/2.1.0/files/p1[0-3]/*/s*/*.jpg"

RUNS_ROOT   = Path("/home/jungsh210/Chest_xray/code/CoT_Web/Medical/backend/runs")
RUNS_ROOT.mkdir(parents=True, exist_ok=True)

EVAL1_ROOT = Path("/home/jungsh210/Chest_xray/code/CoT_Web/Medical/backend/train/outputs_eval1")
EVAL2_ROOT = Path("/home/jungsh210/Chest_xray/code/CoT_Web/Medical/backend/train/outputs_eval2")
EVAL1_ROOT.mkdir(parents=True, exist_ok=True)
EVAL2_ROOT.mkdir(parents=True, exist_ok=True)

LOGS_DIR = RUNS_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
