import os
import numpy as np

_AGE_MODEL = None
_GENDER_MODEL = None
_VIEW_MODEL = None
_INIT_DONE = False

AGE_MODEL_PATH = "/home/jungsh210/Chest_xray/code/CoT_Web/Medical/backend/model/age_model.keras"
GENDER_MODEL_PATH = "/home/jungsh210/Chest_xray/code/CoT_Web/Medical/backend/model/gender_model.keras"
VIEW_MODEL_PATH = "/home/jungsh210/Chest_xray/code/CoT_Web/Medical/backend/model/view_model_best.keras"

def _lazy_init():
    global _AGE_MODEL, _GENDER_MODEL, _VIEW_MODEL, _INIT_DONE
    if _INIT_DONE:
        return
    try:
        import tensorflow as tf
        try:
            tf.config.set_visible_devices([], "GPU")  
        except Exception:
            pass
        from tensorflow.keras.models import load_model
        if os.path.exists(AGE_MODEL_PATH):
            _AGE_MODEL = load_model(AGE_MODEL_PATH)
            print("[INFO] age_model loaded")
        if os.path.exists(GENDER_MODEL_PATH):
            _GENDER_MODEL = load_model(GENDER_MODEL_PATH)
            print("[INFO] gender_model loaded")
        if os.path.exists(VIEW_MODEL_PATH):
            _VIEW_MODEL = load_model(VIEW_MODEL_PATH)
            print("[INFO] view_model loaded")
    except Exception as e:
        print("[WARN] TF/Keras init failed:", e)
    _INIT_DONE = True

def _prep_tensor(image_path: str, size=(224,224)):
    import tensorflow as tf
    img = tf.keras.utils.load_img(image_path, target_size=size)
    arr = tf.keras.utils.img_to_array(img)
    arr = tf.expand_dims(arr, 0)  # (1,H,W,C)
    arr = arr / 255.0
    return arr

def predict_age_gender(image_path: str):
    _lazy_init()
    age = None
    sex = None
    try:
        if _AGE_MODEL is not None:
            arr = _prep_tensor(image_path)
            a = float(_AGE_MODEL.predict(arr, verbose=0)[0][0])
            # 합리적 범위로 클리핑
            age = round(float(np.clip(a, 0.0, 110.0)), 1)
        if _GENDER_MODEL is not None:
            arr = _prep_tensor(image_path)
            p = float(_GENDER_MODEL.predict(arr, verbose=0)[0][0])
            sex = "M" if p >= 0.5 else "F"
    except Exception as e:
        print("[WARN] age/sex predict failed:", e)
    return age, sex

def predict_view(image_path: str):
    _lazy_init()
    if _VIEW_MODEL is None:
        return None
    try:
        arr = _prep_tensor(image_path)
        pred = _VIEW_MODEL.predict(arr, verbose=0)[0]
        idx = int(np.argmax(pred))
        if idx == 0:
            return "AP"
        elif idx == 2:
            return "PA"
        else:
            return None
    except Exception as e:
        print("[WARN] view predict failed:", e)
        return None
