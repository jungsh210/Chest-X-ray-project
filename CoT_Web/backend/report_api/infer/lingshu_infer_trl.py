import os, re
from typing import Dict, Tuple, Optional

import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    GenerationConfig,
)
from peft import PeftModel

_MODEL_CACHE: Dict[Tuple[str, str], Dict] = {}


# ---------------- Utils ----------------
def _postprocess(raw: str) -> str:
    """
    TRL 출력에서 Findings / Impression 외의 부분은 최대한 제거하고,
    두 섹션만 깔끔하게 남기기 위한 후처리.
    """
    if not raw:
        return ""

    txt = raw.strip()

    # 일단 줄바꿈/공백 정리
    txt = txt.replace("\r\n", "\n").replace("\r", "\n")

    # 대소문자 무시하고 Findings, Impression 블록만 추출
    # 패턴:
    #   ... Findings: (여러 줄) Impression: (여러 줄) ...
    pattern = re.compile(
        r"(?is)"          # ignore case + dot matches newline
        r"findings\s*:?\s*(.*?)"   # group 1: Findings 내용
        r"(?:\n+|\s+)*"            # 중간 공백
        r"impression\s*:?\s*(.*)", # group 2: Impression 전체
    )

    m = pattern.search(txt)
    if m:
        findings_body = m.group(1).strip()
        impression_block = m.group(2).strip()

        # Impression 블록에서 다시 'Impression' 헤더 제거
        m2 = re.search(r"(?is)^impression\s*:?\s*(.*)", impression_block)
        if m2:
            impression_body = m2.group(1).strip()
        else:
            impression_body = impression_block.strip()

        out = "Findings:\n" + findings_body + "\n\nImpression:\n" + impression_body
    else:
        # 혹시 위 패턴이 안 맞는 경우 기존 텍스트에서 "Findings"/"Impression"만 정리
        txt2 = re.sub(r"\bFindings\s*:\s*", "Findings:\n", txt, flags=re.I)
        txt2 = re.sub(r"\bImpression\s*:\s*", "\n\nImpression:\n", txt2, flags=re.I)
        out = txt2

    # 오타/표현 정리
    out = out.replace("Indeterminant", "Indeterminate").replace("cardiomeagly", "cardiomegaly")

    return out.strip()


def _safe_decode_ids(processor, ids_tensor) -> str:
    """
    Qwen2 토크나이저에서 None 토큰이 끼어들어오는 경우를 방지하기 위한
    안전 디코더. None 은 건너뛰고 문자열로만 구성해서 붙인다.
    """
    tk = processor.tokenizer
    ids = ids_tensor.tolist()

    # ids -> 토큰으로 변환
    tokens = []
    for i in ids:
        try:
            tok = tk._convert_id_to_token(int(i))
        except Exception:
            tok = None
        if tok is None:
            continue
        tokens.append(tok)

    # 혹시 모를 None 섞임 방지
    tokens = [t for t in tokens if isinstance(t, str)]

    try:
        text = tk.convert_tokens_to_string(tokens)
    except TypeError:
        # 여기가 또 터지면 그냥 안전하게 join
        text = "".join(t for t in tokens if isinstance(t, str))

    return text


# ---------------- Model loader ----------------
def _load_model(base_model_path: str, lora_path: Optional[str] = None):
    """
    base_model_path: TRL로 미세조정된 베이스(또는 동일 베이스)
    lora_path: TRL(+LoRA) 어댑터 경로 (adapter_model.safetensors)
    """
    key = (base_model_path, lora_path or "")
    if key in _MODEL_CACHE:
        obj = _MODEL_CACHE[key]
        return obj["model"], obj["processor"]

    if torch.cuda.device_count() == 0:
        raise RuntimeError("No CUDA device visible. Check CUDA_VISIBLE_DEVICES.")

    max_memory = {i: "22GiB" for i in range(torch.cuda.device_count())}
    max_memory["cpu"] = "120GiB"
    os.makedirs("./offload_cache", exist_ok=True)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        attn_implementation=None,
        device_map="auto",
        low_cpu_mem_usage=True,
        max_memory=max_memory,
        offload_folder="./offload_cache",
        trust_remote_code=True,
    )

    processor = AutoProcessor.from_pretrained(
        base_model_path, use_fast=False, trust_remote_code=True
    )
    tok = processor.tokenizer
    if getattr(tok, "pad_token_id", None) is None:
        tok.pad_token = tok.eos_token or "</s>"

    if lora_path:
        adapter_fp = os.path.join(lora_path, "adapter_model.safetensors")
        if os.path.exists(adapter_fp):
            print(f"[Info] Loading TRL+LoRA from: {lora_path}")
            model = PeftModel.from_pretrained(model, lora_path)
            try:
                model = model.merge_and_unload()
                print("[Info] LoRA merged into base.")
            except Exception:
                print("[Info] Using LoRA adapter without merging.")
        else:
            print(f"[WARN] adapter_model.safetensors not found in {lora_path}")

    model.generation_config = GenerationConfig(
        do_sample=True,
        temperature=0.2,
        top_p=0.3,
        repetition_penalty=1.1,
        max_new_tokens=220,
        min_new_tokens=60,
        no_repeat_ngram_size=3,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.pad_token_id
        or processor.tokenizer.eos_token_id,
    )

    _MODEL_CACHE[key] = {"model": model, "processor": processor}
    return model, processor


# ---------------- Prompt builder ----------------
def _build_refine_prompt(
    prev_report: str,
    view: Optional[str] = None,
    age: Optional[float] = None,
    sex: Optional[str] = None,
    style_hint: Optional[str] = None,
    cot_hint: Optional[str] = None,
) -> str:
    """
    TRL 학습 시 사용한 포맷에 맞춰 '개선 지시문'을 구성.
    prev_report: 9001(LoRA) 1차 결과
    view/age/sex: 선택적 메타
    style_hint: 어투/형식 가이드
    cot_hint: 선택적 reasoning 힌트
    """
    meta_lines = []
    if view:
        meta_lines.append(f"- View: {view}")
    if age is not None:
        meta_lines.append(f"- Age: {age}")
    if sex:
        meta_lines.append(f"- Sex: {sex}")
    meta_txt = "\n".join(meta_lines) if meta_lines else " - (no extra meta)"

    style_txt = style_hint or "Keep it concise and radiology-standardized."

    user_text = (
        "You are a radiology report assistant. Improve the following draft report.\n"
        "Requirements:\n"
        f"{style_txt}\n"
        "\n"
        "Meta:\n"
        f"{meta_txt}\n"
        "\n"
        "Draft report:\n"
        f"{prev_report.strip()}\n"
        "\n"
        "Return ONLY the final report with sections 'Findings' and 'Impression'."
    )
    if cot_hint:
        user_text += f"\n\nHint:\n{cot_hint.strip()}\n"

    return user_text

def clean_trl_report(raw: str) -> str:
    """
    TRL 모델이 JSON/이모지/코드블록까지 같이 내보내는 경우가 있어서
    Findings / Impression 부분만 깔끔하게 남겨주는 후처리 함수.
    """
    if raw is None:
        return ""

    text = raw.strip()

    # 1) ``` 로 시작하는 코드블록 이후(예: ```json ...)는 다 버림
    if "```" in text:
        text = text.split("```", 1)[0].strip()

    # 2) 흔한 JSON 시작 패턴이 있으면 그 이전까지만 사용
    for marker in ['\n{', '\n[', '\n  {', '\n  [']:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx].strip()
            break

    # 3) Findings / Impression 두 섹션만 남기기
    #    예: "Findings: ... Impression: ..." 이런 형태라고 가정
    m = re.search(r"(Findings:\s*.*?)(Impression:\s*.*)", text, re.S | re.I)
    if m:
        text = (m.group(1) + "\n" + m.group(2)).strip()

    # 4) 맨 끝에 남은 이상한 이모지/기호들은 대략 제거
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E]", " ", text)   # ASCII만 남기기
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n\s+", "\n", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text.strip()

def run_lingshu_trl_textonly(
    model_path: str,
    prev_report: str,
    view: Optional[str] = None,
    age: Optional[float] = None,
    sex: Optional[str] = None,
    lora_path: Optional[str] = None,
    style_hint: Optional[str] = None,
    cot_hint: Optional[str] = None,
) -> str:
    """
    텍스트 전용 TRL(+LoRA) 인퍼런스.
    - 이미지 입력 없음
    - prev_report(1차 보고서)를 기반으로 개선본 생성
    """
    model, processor = _load_model(model_path, lora_path)

    user_text = _build_refine_prompt(
        prev_report=prev_report,
        view=view,
        age=age,
        sex=sex,
        style_hint=style_hint,
        cot_hint=cot_hint,
    )

    messages = [{"role": "user", "content": user_text}]
    chat = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[chat],
        return_tensors="pt",
        padding="longest",
        truncation=True,  # 텍스트-only라 컨텍스트 보호용
        max_length=32768,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs)

    # 입력 길이만큼 자르고 생성 부분만 디코딩
    gen_ids_trim = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], outputs)
    ]

    try:
        text = processor.batch_decode(
            gen_ids_trim,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
    except TypeError:
        print("[WARN] batch_decode TypeError 발생, safe decode로 fallback")
        text = _safe_decode_ids(processor, gen_ids_trim[0])

    # 1차: 코드블록/JSON/이모지 제거
    text = clean_trl_report(text)
    # 2차: Findings / Impression 구조만 남기고 오타 보정
    return _postprocess(text)