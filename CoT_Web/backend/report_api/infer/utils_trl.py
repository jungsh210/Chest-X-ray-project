import os, json
import numpy as np
from typing import Optional, Dict
from PIL import Image
import pandas as pd

# ====== (기존) 유틸 ======
def parse_ids_from_path(image_path: str):
    parts = image_path.split("/")
    subject_id, study_id = None, None
    for i, p in enumerate(parts):
        if p.startswith("p") and len(p) == 9 and p[1:].isdigit():
            subject_id = p[1:]
            if i + 1 < len(parts) and parts[i + 1].startswith("s") and parts[i + 1][1:].isdigit():
                study_id = parts[i + 1][1:]
    filename = os.path.basename(image_path)
    dicom_id = os.path.splitext(filename)[0]
    return subject_id, study_id, dicom_id

def load_original_report(image_path: str, report_root: str):
    sid, stid, _ = parse_ids_from_path(image_path)
    if not sid or not stid:
        return None
    report_file = os.path.join(report_root, f"p{sid[:2]}", f"p{sid}", f"s{stid}.txt")
    if not os.path.exists(report_file):
        return None
    with open(report_file, "r") as f:
        return f.read()

def filter_images_by_view(image_paths, metadata_csv: str, allowed_views={"AP", "PA"}):
    """
    (평가 파이프라인 유지용) CSV 기반 AP/PA 필터. DPO 학습 프롬프트에는 CSV를 쓰지 않음.
    """
    allowed = set([v.upper() for v in allowed_views])
    out = []
    usecols = ['subject_id', 'study_id', 'dicom_id', 'ViewPosition']
    df = pd.read_csv(metadata_csv, usecols=usecols)
    df['ViewPosition'] = df['ViewPosition'].str.upper().str.strip()
    idx = {(int(r.subject_id), int(r.study_id), str(r.dicom_id)): r.ViewPosition for _, r in df.iterrows()}
    for img in image_paths:
        sid, stid, did = parse_ids_from_path(img)
        if not sid or not stid or not did:
            continue
        key = (int(sid), int(stid), did)
        vp = idx.get(key, None)
        if vp in allowed:
            out.append(img)
    return out

def build_prompt_strong(meta: dict | None) -> str:
    meta_bits = []
    if meta:
        if meta.get("sex"): meta_bits.append(f"sex={meta['sex']}")
        if meta.get("age_years") is not None: meta_bits.append(f"age≈{meta['age_years']}y")
        if meta.get("view"): meta_bits.append(f"projection={meta['view']}")
    meta_line = f"Patient Context: {', '.join(meta_bits)}.\n" if meta_bits else ""
    return (
        "You are an experienced chest X-ray radiologist.\n\n"
        + meta_line +
        "TASK\nWrite a professional chest X-ray report for the given image.\n\n"
        "OUTPUT FORMAT — use EXACTLY this structure and section titles:\n"
        "Findings:\n"
        "- Projection/quality/rotation: <text>\n"
        "- Devices/lines (ETT/NG/PICC/pacer/wires): <text>\n"
        "- Lungs (volumes, edema/congestion): <text>\n"
        "- Parenchyma nodules: Present | Absent | Indeterminate\n"
        "- Consolidation/airspace opacity: Present | Absent | Indeterminate\n"
        "- Interstitial pattern/reticulation: Present | Absent | Indeterminate\n"
        "- Pleural effusion: Present | Absent | Indeterminate\n"
        "- Pneumothorax: Present | Absent | Indeterminate\n"
        "- Cardiomediastinum/hila: <text>\n\n"
        "Impression:\n"
        "- 1–3 bullet points summarizing key positives and their clinical significance.\n"
        "- If visibility is limited, say 'limited evaluation'.\n\n"
        "CONSTRAINTS\n"
        "- Keep Findings concise; avoid repetition.\n"
        "- Use the exact labels above; do not add extra sections.\n"
        "- For categorical items, default to 'Absent' unless clearly visible; use 'Indeterminate' only if equivocal.\n"
        "- Do not fabricate clinical history.\n"
    )

def build_prompt_existing(meta: dict | None) -> str:
    head = []
    if meta:
        if meta.get("view"): head.append(f"Projection: {meta['view']}")
        if meta.get("sex"): head.append(f"Sex: {meta['sex']}")
        if meta.get("age_years") is not None: head.append(f"Age≈{meta['age_years']}")
    meta_line = f"({'; '.join(head)})\n\n" if head else ""
    detailed_prompt_body = (
        "You are a board-certified cardiothoracic radiologist. Analyze the provided chest X-ray image and write a concise, structured report. \n"
        "Return exactly two sections: \n"
        "Findings: \n - Objective image findings only. \n"
        "Impression: \n - Short, prioritized diagnostic impressions.\n "
        "Avoid extra headers. Use clear clinical English."
    )
    return meta_line + detailed_prompt_body

# ====== (신규) Keras 기반 메타 예측기 ======
class KerasMetaPredictor:
    """
    - TF는 CPU 강제 (GPU 메모리 점유 방지)
    - age_model: 회귀(1) 또는 분류(n) 지원
    - gender_model: 시그모이드(1) 또는 소프트맥스(2) 지원 → 'Male'/'Female'
    - view_model: 시그모이드(1: AP/PA) 또는 소프트맥스(2: AP/PA)
    """
    def __init__(self, age_model: str, gender_model: str, view_model: str, img_size: int = 224):
        try:
            import tensorflow as tf
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
        self.img_size = img_size
        self._age = self._safe_load(age_model)
        self._gender = self._safe_load(gender_model)
        self._view = self._safe_load(view_model)

    def _safe_load(self, path: str):
        if not path or not os.path.exists(path): return None
        try:
            from tensorflow import keras
            return keras.models.load_model(path)
        except Exception:
            return None

    def _load_img_rgb(self, path: str) -> Optional[np.ndarray]:
        try:
            from PIL import Image
            img = Image.open(path).convert("RGB").resize((self.img_size, self.img_size), Image.BILINEAR)
            return np.asarray(img, dtype=np.float32) / 255.0
        except Exception:
            return None

    def predict_age(self, img_path: str) -> Optional[int]:
        if self._age is None: return None
        arr = self._load_img_rgb(img_path)
        if arr is None: return None
        import numpy as np
        y = self._age.predict(arr[None, ...], verbose=0)
        y = np.asarray(y)
        if y.ndim == 2 and y.shape[1] == 1:
            age = float(y[0, 0])
            if not np.isfinite(age): return None
            return int(round(np.clip(age, 0, 120)))
        elif y.ndim == 2 and y.shape[1] > 1:
            idx = int(np.argmax(y[0]))
            return int(idx)
        return None

    def predict_gender(self, img_path: str) -> Optional[str]:
        if self._gender is None: return None
        arr = self._load_img_rgb(img_path)
        if arr is None: return None
        import numpy as np
        y = self._gender.predict(arr[None, ...], verbose=0)
        y = np.asarray(y)
        if y.ndim == 2 and y.shape[1] == 1:
            p = float(y[0, 0])
            if not np.isfinite(p): return None
            return "Male" if p >= 0.5 else "Female"
        elif y.ndim == 2 and y.shape[1] == 2:
            idx = int(np.argmax(y[0]))
            return "Male" if idx == 1 else "Female"
        return None

    def predict_view(self, img_path: str) -> Optional[str]:
        if self._view is None: return None
        arr = self._load_img_rgb(img_path)
        if arr is None: return None
        import numpy as np
        y = self._view.predict(arr[None, ...], verbose=0)
        y = np.asarray(y)
        if y.ndim == 2 and y.shape[1] == 1:
            p = float(y[0, 0])
            if not np.isfinite(p): return None
            return "PA" if p >= 0.5 else "AP"
        elif y.ndim == 2 and y.shape[1] >= 2:
            idx = int(np.argmax(y[0]))
            if y.shape[1] == 2:
                return "PA" if idx == 1 else "AP"
            return None
        return None
    
def load_view_from_meta(image_path: str, metadata_csv: str):
    """
    MIMIC-CXR metadata CSV에서 해당 이미지의 ViewPosition(AP/PA)만 조회.
    - lingshu_infer.py가 use_meta=True일 때 호출함.
    """
    sid, stid, did = parse_ids_from_path(image_path)
    if not sid or not stid or not did:
        return None

    usecols = ['subject_id', 'study_id', 'dicom_id', 'ViewPosition']
    try:
        # 큰 CSV를 안전하게 순회
        for chunk in pd.read_csv(metadata_csv, usecols=usecols, chunksize=200000):
            hit = chunk[
                (chunk['subject_id'] == int(sid)) &
                (chunk['study_id'] == int(stid)) &
                (chunk['dicom_id'] == did)
            ]
            if len(hit) > 0:
                vp = str(hit.iloc[0]['ViewPosition']).upper().strip()
                return vp if vp in {"AP", "PA"} else None
    except Exception:
        # CSV 없거나 포맷 이슈 시 None
        return None

    return None

def build_prompt_dpo(meta: Dict, first_gen: str, physician_feedback: Optional[str], corrected_report: Optional[str]) -> str:
    """
    DPO 학습 컨텍스트: 이미지 + 메타 + (1차 출력, 피드백, 수정판독문)를 프롬프트에 포함
    """
    bits = []
    if meta.get("view"): bits.append(f"Projection: {meta['view']}")
    if meta.get("sex"): bits.append(f"Sex: {meta['sex']}")
    if meta.get("age_years") is not None: bits.append(f"Age≈{meta['age_years']}y")

    fb = (physician_feedback or "").strip()
    corr = (corrected_report or "").strip()

    parts = []
    if bits:
        parts.append("(" + "; ".join(bits) + ")\n")
    parts.append("You are a cardiothoracic radiologist. Improve the draft chest X-ray report using the image and clinical feedback.\n\n")
    parts.append("DRAFT (first-pass model):\n")
    parts.append(first_gen.strip() + "\n\n")
    if fb:
        parts.append("PHYSICIAN FEEDBACK:\n" + fb + "\n\n")
    if corr:
        parts.append("REVISED REPORT (gold):\n" + corr + "\n\n")
    parts.append("TASK: Write a concise, structured report with exactly two sections 'Findings' and 'Impression'. "
                 "Avoid extra headers. Use clear clinical English.")

    return "".join(parts)