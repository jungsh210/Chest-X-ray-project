import os, re
from typing import Dict, Tuple, Optional

from PIL import Image
import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    GenerationConfig,
)
from peft import PeftModel

from .qwen_vl_utils import process_vision_info
from .utils import (
    build_prompt_strong,
    build_prompt_existing,
    load_view_from_meta,
)
from .utils_trl import KerasMetaPredictor

_MODEL_CACHE: Dict[Tuple[str, str], Dict] = {}  # key: (base_model_path, lora_path or "")
_DEM_CACHE: Dict[str, Tuple[Optional[float], Optional[str]]] = {}

# ---------------------------
# (선택) 나이/성별 예측기
# ---------------------------
_AGE_GENDER_INIT = False
_AGE_MODEL = None
_GENDER_MODEL = None
_AGE_MODEL_PATH = "/home/jungsh210/Chest_xray/code/1103/model/age_model.keras"
_GENDER_MODEL_PATH = "/home/jungsh210/Chest_xray/code/1103/model/gender_model.keras"


def _init_age_gender_models():
    """TensorFlow 로드 및 Keras 모델 로딩(실패해도 전체 파이프라인은 계속 진행)"""
    global _AGE_GENDER_INIT, _AGE_MODEL, _GENDER_MODEL
    if _AGE_GENDER_INIT:
        return
    try:
        import tensorflow as tf
        try:
            # TF가 GPU 메모리 점유하지 않도록 CPU 고정
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass

        from tensorflow.keras.models import load_model
        if os.path.exists(_AGE_MODEL_PATH) and os.path.exists(_GENDER_MODEL_PATH):
            _AGE_MODEL = load_model(_AGE_MODEL_PATH)
            _GENDER_MODEL = load_model(_GENDER_MODEL_PATH)
        else:
            print("[Ctx] age_model.keras / gender_model.keras not found. Proceeding without demographics.")
    except Exception as e:
        print("[TF] TensorFlow import or loading failed:", e)
    _AGE_GENDER_INIT = True


def _predict_age_gender(image_path: str) -> Tuple[Optional[float], Optional[str]]:
    """이미지에서 (age_years, sex[M/F]) 추정. 실패 시 (None, None)"""
    if image_path in _DEM_CACHE:
        return _DEM_CACHE[image_path]
    _init_age_gender_models()
    if _AGE_MODEL is None or _GENDER_MODEL is None:
        _DEM_CACHE[image_path] = (None, None)
        return (None, None)
    try:
        import tensorflow as tf
        IMG_HEIGHT, IMG_WIDTH = 224, 224
        img = tf.keras.utils.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        arr = tf.keras.utils.img_to_array(img)
        arr = tf.expand_dims(arr, 0)
        age_pred = float(_AGE_MODEL.predict(arr, verbose=0)[0][0])
        gender_prob = float(_GENDER_MODEL.predict(arr, verbose=0)[0][0])
        sex = "M" if gender_prob >= 0.5 else "F"
        age_years = round(age_pred, 1)
        _DEM_CACHE[image_path] = (age_years, sex)
        return (age_years, sex)
    except Exception as e:
        print(f"[TF] Age/Gender inference failed for {image_path}: {e}")
        _DEM_CACHE[image_path] = (None, None)
        return (None, None)


# ---------------------------
# 모델 로더 (베이스 + 옵션 LoRA)
# ---------------------------
def _load_model(base_model_path: str, lora_path: Optional[str] = None):
    """
    base_model_path: Lingshu-7B or 32B (Qwen2.5-VL) 경로
    lora_path: LoRA adapter 디렉토리 (adapter_model.safetensors 포함). 없으면 베이스만 사용.
    """
    cache_key = (base_model_path, lora_path or "")
    if cache_key in _MODEL_CACHE:
        obj = _MODEL_CACHE[cache_key]
        return obj["model"], obj["processor"]

    n = torch.cuda.device_count()
    if n == 0:
        raise RuntimeError("No CUDA device visible. Check CUDA_VISIBLE_DEVICES.")

    # 보이는 장치 기준 메모리 한도 설정 (필요에 맞게 조절)
    max_memory = {i: "22GiB" for i in range(n)}
    max_memory["cpu"] = "120GiB"
    os.makedirs("./offload_cache", exist_ok=True)

    # 1) 베이스 모델 로드
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,           # 추론 메모리 절감
        attn_implementation=None,
        device_map="auto",
        low_cpu_mem_usage=True,
        max_memory=max_memory,
        offload_folder="./offload_cache",
    )

    # 2) Processor (tokenizer 포함)
    #  - LoRA 폴더에 tokenizer/chat_template가 있으면 우선 사용
    proc_src = base_model_path
    if lora_path and os.path.exists(os.path.join(lora_path, "tokenizer_config.json")):
        proc_src = lora_path
    processor = AutoProcessor.from_pretrained(proc_src, use_fast=True)

    # pad_token 안정화
    tok = processor.tokenizer
    if getattr(tok, "pad_token_id", None) is None:
        tok.pad_token = tok.eos_token or "</s>"

    # 3) LoRA 어댑터 로드(선택)
    if lora_path:
        adapter_fp = os.path.join(lora_path, "adapter_model.safetensors")
        if os.path.exists(adapter_fp):
            print(f"[Info] Loading LoRA adapter from: {lora_path}")
            model = PeftModel.from_pretrained(model, lora_path)
            # 메모리 여유 시 병합, 실패하면 어댑터 상태로 그대로 사용
            try:
                model = model.merge_and_unload()
                print("[Info] LoRA weights merged into base model.")
            except Exception:
                print("[Info] Using LoRA adapter without merging.")
        else:
            print(f"[WARN] LoRA adapter not found in: {lora_path} (skip)")

    # 4) 생성 설정
    gen_cfg = GenerationConfig(
        do_sample=True,
        temperature=0.2,
        top_p=0.3,
        repetition_penalty=1.1,
        max_new_tokens=200,
        min_new_tokens=80,
        no_repeat_ngram_size=3,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
    )
    model.generation_config = gen_cfg

    _MODEL_CACHE[cache_key] = {"model": model, "processor": processor}
    return model, processor


# ---------------------------
# 후처리
# ---------------------------
def _postprocess(txt: str) -> str:
    txt = re.sub(r"\bFindings\s*:\s*", "Findings:\n", txt, flags=re.I)
    txt = re.sub(r"\bImpression\s*:\s*", "\n\nImpression: ", txt, flags=re.I)
    txt = txt.replace("Indeterminant", "Indeterminate").replace("cardiomeagly", "cardiomegaly")
    return txt


# ---------------------------
# 공개 API
# ---------------------------
def run_lingshu(
    model_path: str,
    image_path: str,
    prompt_type: str,
    use_meta: bool,
    meta_csv: str,
    lora_path: Optional[str] = None,
) -> str:
    """
    Lingshu-7B/32B (Qwen2.5-VL) + (옵션) LoRA로 CXR 보고서 생성
    - prompt_type: "Strong" | "Existing"
    - use_meta: True면 AP/PA + (가능 시) age/sex 메타를 포함해 프롬프트 구성
    - lora_path: LoRA 어댑터 폴더 경로 또는 None
    """
    model, processor = _load_model(base_model_path=model_path, lora_path=lora_path)

    # 메타 구성
    meta = {}
    if use_meta:
        # CSV 대신 Keras 예측 사용 원하시면:
        predictor = KerasMetaPredictor(
            age_model="/home/jungsh210/Chest_xray/code/1104-2/model/age_model.keras",
            gender_model="/home/jungsh210/Chest_xray/code/1104-2/model/gender_model.keras",
            view_model="/home/jungsh210/Chest_xray/code/1104-2/model/view_model_best.keras",
            img_size=224
        )
        age_years = predictor.predict_age(image_path)
        sex = predictor.predict_gender(image_path)
        view = predictor.predict_view(image_path)
        if age_years is not None: meta["age_years"] = age_years
        if sex: meta["sex"] = sex
        if view: meta["view"] = view

    prompt = build_prompt_strong(meta) if prompt_type == "Existing" else build_prompt_existing(meta)

    # 입력 구성
    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[chat],
        images=image_inputs,
        videos=video_inputs,
        padding="max_length",
        truncation=False,
        return_tensors="pt",
    ).to(model.device)

    # 생성 (OOM 대비 2단계 fallback)
    try:
        with torch.no_grad():
            generated = model.generate(**inputs)
    except RuntimeError as e:
        if "CUDA out of memory" not in str(e):
            raise
        torch.cuda.empty_cache()
        try:
            with torch.no_grad():
                generated = model.generate(**inputs, use_cache=False, max_new_tokens=160)
        except RuntimeError as e2:
            if "CUDA out of memory" in str(e2):
                torch.cuda.empty_cache()
                return ""  # 스킵 -> 평가 시 자동 제외
            raise

    # 입력 프롬프트 길이만큼 잘라서 생성 토큰만 디코딩
    gen_ids_trim = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated)]
    text = processor.batch_decode(gen_ids_trim, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return _postprocess(text)