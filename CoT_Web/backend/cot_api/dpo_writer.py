import json, os

def append_feedback_line(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def append_dpo_pair(path: str, *, prompt: str, chosen: str, rejected: str, image_path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rec = {
        "prompt": prompt,
        "chosen": (chosen or "").strip(),
        "rejected": (rejected or "").strip(),
        "image": image_path,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def build_prompt_existing_with_feedback(meta: dict | None,
                                        clinician_comment: str | None = None,
                                        cot: dict | None = None) -> str:
    head = []
    if meta:
        if meta.get("view"): head.append(f"Projection: {meta['view']}")
        if meta.get("sex"): head.append(f"Sex: {meta['sex']}")
        if meta.get("age_years") is not None: head.append(f"Age≈{meta['age_years']}")
    meta_line = f"({'; '.join(head)})\n\n" if head else ""

    notes = []
    if cot:
        if cot.get("first_look"):     notes.append(f"FIRST LOOK (scan path): {cot['first_look']}")
        if cot.get("not_on_image"):   notes.append(f"NOT ON IMAGE (normal/clear): {cot['not_on_image']}")
        if cot.get("uncertainty"):    notes.append(f"UNCERTAINTY: {cot['uncertainty']}")
        if cot.get("final_diagnosis"):notes.append(f"FINAL DIAGNOSIS (with why): {cot['final_diagnosis']}")
        if cot.get("next_step"):      notes.append(f"NEXT STEP (tests/markers): {cot['next_step']}")
    if clinician_comment:
        notes.append(f"NOTE from clinician: {clinician_comment.strip()}")

    fb_block = ""
    if notes:
        fb_block = "=== CLINICIAN REASONING NOTES ===\n" + "\n".join(f"- {n}" for n in notes) + "\n\n"

    body = (
        "You are a board-certified cardiothoracic radiologist. "
        "Analyze the provided chest X-ray image and write a concise, structured report.\n"
        "Return exactly two sections:\n"
        "Findings:\n - Objective image findings only.\n"
        "Impression:\n - Short, prioritized diagnostic impressions.\n"
        "Avoid extra headers. Use clear clinical English."
    )
    return meta_line + fb_block + body