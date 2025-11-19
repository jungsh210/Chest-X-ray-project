# backend/utils/csv_logger.py
import csv
import os
from pathlib import Path
from typing import Dict

COT_CSV_PATH = Path("cot_feedback.csv")

COLUMNS = [
    "case_id",
    "train_image_path",
    "user_image_path",
    "cot_text",
    "lora_report",
    "trl_report",
    "clinical_score",
    "expression_score",
    "logic_score",
    "better_model",
    "comment",
]

def append_row(data: Dict):
    file_exists = COT_CSV_PATH.exists()

    with COT_CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if not file_exists:
            writer.writeheader()

        row = {col: data.get(col, "") for col in COLUMNS}
        writer.writerow(row)
