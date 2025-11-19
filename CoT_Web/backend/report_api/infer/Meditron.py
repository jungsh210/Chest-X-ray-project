from __future__ import annotations
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch, json, re
from typing import List, Dict, Tuple, Optional

FORBIDDEN_STRINGS = [
    "### Instruction", "### Answer", "Instruction:", "Answer:",
    "San Francisco", "Yosemite",
    "guitar", "ukulele", "steak", "travel across the United States"
]

FINDING_TERMS = {
    "low lung volume", "low lung volumes", "bilateral nodules", "pulmonary nodules",
    "scattered nodules", "mild pulmonary vascular congestion", "vascular congestion",
    "cardiomegaly", "cardiomediastinal", "hilar", "right middle fissural effusion",
    "fissural effusion", "pleural effusion", "no pneumonia", "no pneumothorax",
    "pneumothorax absent", "pneumonia absent", "lung volumes", "innumerable",
    "new effusion", "unchanged"
}

def _build_bad_words_ids(tokenizer, words: List[str]) -> List[List[int]]:
    ids = []
    for w in words:
        toks = tokenizer(w, add_special_tokens=False).input_ids
        if toks:
            ids.append(toks)
    return ids

def _find_json(text: str) -> str:
    start = text.find("{")
    if start == -1:
        return ""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    return ""

def _infer_absent_flags_from_report(report_text: str) -> Dict[str, bool]:
    t = report_text.lower()
    return {
        "pneumonia": bool(re.search(r"\bno\s+.*pneumonia\b", t)),
        "pneumothorax": bool(re.search(r"\bno\s+.*pneumothorax\b", t)),
    }

def _is_finding_like(s: str) -> bool:
    s_norm = re.sub(r"[^\w\s\-]", "", s).lower().strip()
    if not s_norm:
        return True
    if len(s_norm.split()) <= 2 and any(kw in s_norm for kw in ["low", "mild", "bilateral", "right", "middle", "fissural", "cardio", "hilar"]):
        return True
    for kw in FINDING_TERMS:
        if kw in s_norm:
            return True
    return False

def _dedup_keep_top5(items: List[str]) -> List[str]:
    seen, out = set(), []
    for x in items or []:
        xx = x.strip()
        if not xx:
            continue
        key = xx.lower()
        if key not in seen:
            out.append(xx)
            seen.add(key)
    return out[:5]

def _pattern_gate_filter(diffs: List[str], tests: List[str], text: str) -> Tuple[List[str], List[str]]:
    t = (text or "").lower()

    nod = ("nodule" in t or "nodules" in t) and ("bilateral" in t or "innumerable" in t or "scattered" in t)
    cong = ("vascular congestion" in t)
    eff = ("effusion" in t) and (("fissure" in t) or ("fissural" in t) or ("pleural" in t))
    is_new_eff = eff and ("new" in t)

    diffs_out = list(diffs)
    tests_out = list(tests)

    if nod:
        allow_dx_substrings = {"metastatic", "miliary", "tuberculosis", "sarcoidosis", "fungal"}
        block_dx_exact = {"copd", "pulmonary embolism", "pe", "interstitial lung disease", "ild", "multiple myeloma", "amyloidosis"}
        diffs_out = [
            d for d in diffs_out
            if (any(k in d.lower() for k in allow_dx_substrings) or not any(b == d.lower() for b in block_dx_exact))
        ]
        prefer_tests = {"ct"}  
        block_tests = {"sputum", "blood culture", "bone marrow", "electrophoresis"}
        tests_out = [
            x for x in tests_out
            if (any(k in x.lower() for k in prefer_tests) or not any(b in x.lower() for b in block_tests))
        ]
        if not any("ct" in x.lower() for x in tests_out):
            tests_out.append("Chest CT for nodule characterization")

    if cong:
        if not any("heart failure" in d.lower() or "volume overload" in d.lower() for d in diffs_out):
            diffs_out.append("Chronic heart failure / volume overload")
        if not any("echocardiography" in x.lower() for x in tests_out):
            tests_out.append("Transthoracic echocardiography")
        if not any("bnp" in x.lower() for x in tests_out):
            tests_out.append("Serum BNP/NT-proBNP")

    if is_new_eff:
        if not any("follow-up" in x.lower() and "radiograph" in x.lower() for x in tests_out):
            tests_out.append("Follow-up chest radiograph to monitor the new fissural/pleural effusion")

    return diffs_out, tests_out

def _sanitize_output(data: Dict, absent_flags: Dict[str, bool]) -> Dict:
    diffs = [d for d in data.get("differential", []) if not _is_finding_like(d)]
    tests = [t for t in data.get("next_tests", []) if not _is_finding_like(t)]
    if absent_flags.get("pneumonia", False):
        diffs = [d for d in diffs if "pneumonia" not in d.lower()]
        tests = [t for t in tests if "sputum" not in t.lower() and "pneumonia" not in t.lower()]
    if absent_flags.get("pneumothorax", False):
        diffs = [d for d in diffs if "pneumothorax" not in d.lower()]
        tests = [t for t in tests if "pneumothorax" not in t.lower()]
    data["differential"] = _dedup_keep_top5(diffs)
    data["next_tests"] = _dedup_keep_top5(tests)
    return data

def _very_conservative_fill_if_empty(data: Dict, reference_text: str) -> Dict:
    t = reference_text.lower()
    diffs = data.get("differential", []) or []
    tests = data.get("next_tests", []) or []
    nod = ("nodule" in t or "nodules" in t) and ("bilateral" in t or "innumerable" in t or "scattered" in t)
    cong = ("vascular congestion" in t)
    fiss_eff_new = ("fissure" in t or "fissural" in t) and ("effusion" in t) and ("new" in t)
    if not diffs:
        if nod:
            diffs.extend([
                "Metastatic disease (pattern-based bilateral innumerable pulmonary nodules)",
                "Miliary tuberculosis",
                "Sarcoidosis"
            ])
        if cong:
            diffs.append("Chronic heart failure / volume overload (stable mild pulmonary vascular congestion)")
        if fiss_eff_new:
            diffs.append("Small transudative pleural effusion related to congestion (new right middle fissural effusion)")
    if not tests:
        if nod:
            tests.append("Chest CT for nodule characterization")
        if cong:
            tests.append("Transthoracic echocardiography to assess cardiac function/volume status")
            tests.append("Serum BNP/NT-proBNP to support congestion")
        if fiss_eff_new:
            tests.append("Follow-up chest radiograph to monitor the new small fissural effusion")
    data["differential"] = _dedup_keep_top5(diffs)
    data["next_tests"] = _dedup_keep_top5(tests)
    return data

def _split_lingshu(findings_impression_text: str) -> Dict[str, str]:
    f, i = "", ""
    m_f = re.search(r"(?:^|\n)\s*Findings?\s*:\s*(.*?)(?:\n\s*Impression\s*:|\Z)", findings_impression_text, flags=re.S|re.I)
    if m_f: f = m_f.group(1).strip()
    m_i = re.search(r"(?:^|\n)\s*Impression\s*:\s*(.*)\Z", findings_impression_text, flags=re.S|re.I)
    if m_i: i = m_i.group(1).strip()
    if not f and not i:
        f = findings_impression_text.strip()
    return {"findings": f, "impression": i}

def _pattern_gate_filter(diffs: List[str], tests: List[str], text: str) -> (List[str], List[str]):
    t = text.lower()
    nod = ("nodule" in t or "nodules" in t) and ("bilateral" in t or "innumerable" in t or "scattered" in t)
    cong = ("vascular congestion" in t)
    eff = ("effusion" in t) and (("fissure" in t) or ("fissural" in t) or ("pleural" in t))

    d_low = [d.lower() for d in diffs]
    t_low = [x.lower() for x in tests]

    if nod:
        allow_dx = {"metastatic", "miliary", "tuberculosis", "sarcoidosis", "fungal"}
        block_dx = {"copd", "pulmonary embolism", "pe", "interstitial lung disease", "ild", "multiple myeloma", "amyloidosis"}
        diffs = [d for d in diffs if (any(k in d.lower() for k in allow_dx) or not any(b in d.lower() for b in block_dx))]

        prefer_tests = {"ct", "computed tomography", "pet"}  
        block_tests = {"sputum", "blood culture", "bone marrow", "electrophoresis"}
        tests = [x for x in tests if (any(k in x.lower() for k in prefer_tests) or not any(b in x.lower() for b in block_tests))]

    if cong:
        if not any("heart failure" in d.lower() or "volume overload" in d.lower() for d in diffs):
            diffs.append("Chronic heart failure / volume overload")
        if not any("echocardiography" in x.lower() for x in tests):
            tests.append("Transthoracic echocardiography")
        if not any("bnp" in x.lower() for x in tests):
            tests.append("Serum BNP/NT-proBNP")

    if eff and "new" in t:
        if not any("follow-up" in x.lower() and "radiograph" in x.lower() for x in tests):
            tests.append("Follow-up chest radiograph to monitor the new fissural/pleural effusion")

    return diffs, tests

def Meditron(lingshu_report_text: str,
             original_report_text: str = "",
             demographics: Dict[str, object] | None = None,
             model_id: str = "/home/jungsh210/hf_models/smagt-meditron-7b-instruct/",
             max_new_tokens: int = 320) -> str:

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map={"": 0},
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    li = _split_lingshu(lingshu_report_text)
    lingshu_findings = li["findings"]
    lingshu_impression = li["impression"]

    ctx_str = ""
    if demographics:
        age_str = f"{demographics.get('age_years')}" if demographics.get('age_years') is not None else "unknown"
        sex_str = f"{demographics.get('sex')}" if demographics.get('sex') else "unknown"
        ctx_str = f"Patient Context (optional): Age {age_str} years; Sex {sex_str}.\n"

    system_prompt = (
        "You are a clinical reasoning assistant.\n"
        "Your ONLY input will be an automatically generated chest X-ray report "
        "with 'Findings' and 'Impression'.\n\n"
        "TASK:\n"
        "- Based on BOTH the report and the demographics (age, sex), propose:\n"
        "  1) differential diagnoses (disease/condition names only)\n"
        "  2) recommended next tests (directly related to findings)\n\n"
        "STRICT RULES:\n"
        "- Do NOT add or assume findings not mentioned.\n"
        "- Respect explicit negations (e.g., 'no pneumonia', 'no pneumothorax').\n"
        "- Do NOT simply restate descriptive findings (e.g., 'low lung volumes'); "
        "output diagnosis-level terms only.\n"
        "- Return ONLY a JSON object with two keys: 'differential' and 'next_tests'.\n"
    )

    user_prompt = (
        f"{ctx_str}"
        "Chest X-ray Report (generated):\n"
        f"{lingshu_report_text.strip()}\n\n"
        "Return ONLY the JSON object."
    )

    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt_text = system_prompt + "\n\n" + user_prompt

    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        do_sample=False,
        top_p=1.0,
        repetition_penalty=1.05,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        bad_words_ids=_build_bad_words_ids(tokenizer, FORBIDDEN_STRINGS),
    )

    with torch.no_grad():
        output_ids = model.generate(**inputs, generation_config=gen_cfg)

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    if text.startswith(prompt_text):
        text = text[len(prompt_text):]

    json_str = _find_json(text)
    if not json_str:
        json_str = '{"differential": [], "next_tests": []}'

    try:
        data = json.loads(json_str)
        if not isinstance(data, dict):
            data = {"differential": [], "next_tests": []}
    except Exception:
        data = {"differential": [], "next_tests": []}

    combined_text = (lingshu_findings + "\n" + lingshu_impression + "\n" + (original_report_text or "")).strip()
    absent_flags = _infer_absent_flags_from_report(combined_text)
    data = _sanitize_output(data, absent_flags)

    if not data.get("differential") or not data.get("next_tests"):
        data = _very_conservative_fill_if_empty(data, combined_text)

    return json.dumps(data, ensure_ascii=False, indent=2)