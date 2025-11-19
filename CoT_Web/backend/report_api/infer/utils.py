import os
import pandas as pd

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

def load_view_from_meta(image_path: str, metadata_csv: str):
    sid, stid, did = parse_ids_from_path(image_path)
    if not sid or not stid or not did:
        return None
    usecols = ['subject_id', 'study_id', 'dicom_id', 'ViewPosition']
    for chunk in pd.read_csv(metadata_csv, usecols=usecols, chunksize=200000):
        hit = chunk[
            (chunk['subject_id'] == int(sid)) &
            (chunk['study_id'] == int(stid)) &
            (chunk['dicom_id'] == did)
        ]
        if len(hit) > 0:
            vp = str(hit.iloc[0]['ViewPosition']).upper().strip()
            if vp in {"AP", "PA"}:
                return vp
            return None
    return None

def filter_images_by_view(image_paths, metadata_csv: str, allowed_views={"AP", "PA"}):
    """
    이미지 리스트를 순회하며 CSV에서 ViewPosition을 확인.
    allowed_views(AP/PA)에 해당하는 케이스만 반환.
    """
    allowed = set([v.upper() for v in allowed_views])
    out = []
    usecols = ['subject_id', 'study_id', 'dicom_id', 'ViewPosition']
    # 미리 DataFrame 전체를 읽는 대신 딕셔너리 색인으로 최적화 (메모리 여유시)
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
    """
    보편형 강제 포맷 — 메타가 있으면 projection/age/sex 포함
    """
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
    """
    Sets the stage for the Lingshu-7B model, providing detailed instructions
    for generating a structured Chest X-ray report, including patient metadata
    (if available) and the required 'Findings' and 'Impression' sections.
    """
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

