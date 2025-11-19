from pathlib import Path

# TRL 학습에 사용할 베이스 모델 (Lingshu-7B)
BASE_MODEL = "/home/jungsh210/hf_models/lingshu7b_lora/"

# DPO 입력/출력 경로 (cot_api와 동일 장소)
DPO_JSONL = "outputs_cot/dpo_train.jsonl"
OUT_DIR    = "runs/trl_from_cot_v1"      # 학습 결과(LoRA)
FINAL_DIR  = "runs/final_trl_model"      # 최종 배포용 복사본

TRAIN_SCRIPT = "train_trl_offline.py"