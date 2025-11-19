import glob, os, io, json, time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from PIL import Image

from .paths_config import (
    IMAGE_ROOT, FEEDBACK_JSONL, DPO_JSONL, PAGE_SIZE_DEFAULT,
    META_CSV, ALLOWED_VIEWS, PRED_CACHE_JSON, PATIENTS_CSV   
)
from .demographic import predict_age_gender, predict_view
from .dpo_writer import build_prompt_existing_with_feedback, append_feedback_line, append_dpo_pair
from utils.csv_logger import append_row as append_cot_csv_row

from datetime import datetime

app = FastAPI(title="COT API", version="1.5")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

IMAGE_ROOT = Path(IMAGE_ROOT).resolve()
def _abs_to_rel(abs_path: str) -> str:
    p = Path(abs_path).resolve()
    try:
        return str(p.relative_to(IMAGE_ROOT))
    except Exception:
        return str(p)
def _safe_join(rel: str) -> Path:
    p = (IMAGE_ROOT / rel).resolve()
    if not str(p).startswith(str(IMAGE_ROOT)):
        raise HTTPException(status_code=400, detail="Invalid path")
    return p

class PredCache:
    def __init__(self, fp: str):
        self.fp = fp
        self.mem: Dict[str, Dict] = {}
        self._load()
    def _load(self):
        try:
            if os.path.exists(self.fp):
                with open(self.fp, "r", encoding="utf-8") as f:
                    self.mem = json.load(f)
        except Exception:
            self.mem = {}
    def get(self, rel: str) -> Optional[Dict]:
        return self.mem.get(rel)
    def set(self, rel: str, rec: Dict):
        self.mem[rel] = rec
        tmp = self.fp + ".tmp"
        os.makedirs(os.path.dirname(self.fp), exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.mem, f, ensure_ascii=False)  
        os.replace(tmp, self.fp)

PRED_CACHE = PredCache(PRED_CACHE_JSON)

_view_index_cache: Optional[Dict[Tuple[int,int,str], str]] = None
_patient_index_cache: Optional[Dict[int, Dict[str, Optional[str]]]] = None  

def _build_view_index() -> Optional[Dict[Tuple[int,int,str], str]]:
    global _view_index_cache
    if _view_index_cache is not None:
        return _view_index_cache
    if not META_CSV or not os.path.exists(META_CSV):
        _view_index_cache = None
        return None
    import pandas as pd
    usecols = ["subject_id","study_id","dicom_id","ViewPosition"]
    df = pd.read_csv(META_CSV, usecols=usecols)
    df["ViewPosition"] = df["ViewPosition"].astype(str).str.upper().str.strip()
    idx = {(int(r.subject_id), int(r.study_id), str(r.dicom_id)): r.ViewPosition for _, r in df.iterrows()}
    _view_index_cache = idx
    return idx

def _build_patient_index() -> Optional[Dict[int, Dict[str, Optional[str]]]]:
    global _patient_index_cache
    if _patient_index_cache is not None:
        return _patient_index_cache
    if not PATIENTS_CSV or not os.path.exists(PATIENTS_CSV):
        _patient_index_cache = None
        return None
    import pandas as pd
    usecols = ["subject_id", "anchor_age", "gender"]
    df = pd.read_csv(PATIENTS_CSV, usecols=usecols)
    def _map_sex(x: str) -> Optional[str]:
        if not isinstance(x, str): return None
        x = x.strip().upper()
        if x in ("M","MALE"): return "M"
        if x in ("F","FEMALE"): return "F"
        return None
    idx: Dict[int, Dict[str, Optional[str]]] = {}
    for _, r in df.iterrows():
        try:
            sid = int(r.subject_id)
        except Exception:
            continue
        age = None
        try:
            age = float(r.anchor_age)
        except Exception:
            pass
        idx[sid] = {"age_years": (None if age is None else round(age,1)),
                    "sex": _map_sex(str(r.gender))}
    _patient_index_cache = idx
    return idx

def _parse_ids_from_rel(rel: str) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    parts = rel.split("/")
    sid = stid = did = None
    for i, p in enumerate(parts):
        if p.startswith("p") and len(p) == 9 and p[1:].isdigit():
            sid = int(p[1:])
            if i + 1 < len(parts) and parts[i + 1].startswith("s") and parts[i + 1][1:].isdigit():
                stid = int(parts[i + 1][1:])
    if parts:
        did = Path(parts[-1]).stem
    return sid, stid, did

def _lookup_meta_from_csv(rel: str) -> Dict[str, Optional[str]]:
    sid, stid, did = _parse_ids_from_rel(rel)
    age = None; sex = None; view = None

    pidx = _build_patient_index()
    if pidx and sid is not None:
        rec = pidx.get(sid, {})
        age = rec.get("age_years", None)
        sex = rec.get("sex", None)

    vidx = _build_view_index()
    if vidx and sid is not None and stid is not None and did is not None:
        vp = vidx.get((sid, stid, did), None)
        if vp in ALLOWED_VIEWS:
            view = vp
        else:
            view = None

    return {"age_years": age, "sex": sex, "view": view}

class SaveReq(BaseModel):
    rel: str
    edited_report: str
    clinician_comment: Optional[str] = None 
    cot: Optional[Dict[str, Optional[str]]] = None
    age_years: Optional[float] = Field(default=None)
    sex: Optional[str] = Field(default=None)
    view: Optional[str] = Field(default=None)
    first_gen_report: Optional[str] = None

class SaveResp(BaseModel):
    ok: bool
    wrote_feedback: bool
    wrote_dpo: bool

class EvalSaveResp(BaseModel):
    ok: bool

def _gather_rel_images_ap_pa() -> List[str]:
    pats = [str(IMAGE_ROOT / "p1[0-3]" / "*" / "s*" / "*.jpg")]
    files: List[str] = []
    for p in pats:
        files.extend(glob.glob(p))
    files = sorted(set(files))
    rels = [_abs_to_rel(f) for f in files]
    idx = _build_view_index()
    if idx is None:
        return rels
    out: List[str] = []
    for r in rels:
        sid, stid, did = _parse_ids_from_rel(r)
        if sid is None or stid is None or did is None:
            continue
        vp = idx.get((sid, stid, did), "")
        if isinstance(vp, str) and vp.upper() in ALLOWED_VIEWS:
            out.append(r)
    return out

@app.get("/api/cot/list")
def list_images(start: int = Query(0, ge=0), count: int = Query(PAGE_SIZE_DEFAULT, ge=1, le=20)):
    imgs = _gather_rel_images_ap_pa()
    end = min(start + count, len(imgs))
    return {"total": len(imgs), "start": start, "count": end - start, "images": imgs[start:end]}

@app.get("/api/cot/meta")
def get_meta(rel: str = Query(...)):
    _ = _safe_join(rel)
    meta = _lookup_meta_from_csv(rel)
    cur = PRED_CACHE.get(rel) or {}
    cur.update({k: v for k, v in meta.items() if v is not None})
    if cur:
        PRED_CACHE.set(rel, cur)
    return meta

@app.get("/api/cot/image")
def get_image(rel: str = Query(...), w: Optional[int] = Query(None, ge=64, le=4096)):
    img_path = _safe_join(rel)
    if not img_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    if w is None:
        return FileResponse(str(img_path), media_type="image/jpeg")
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        ratio = w / im.width
        h = max(1, int(im.height * ratio))
        im = im.resize((w, h))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=88)
        buf.seek(0)
    return Response(buf.getvalue(), media_type="image/jpeg")

@app.get("/api/cot/predict")
def predict(rel: str = Query(...)):
    cached = PRED_CACHE.get(rel)
    if cached and all(k in cached for k in ("age_years","sex","view")):
        return cached
    _ = _safe_join(rel)
    rec = _lookup_meta_from_csv(rel)
    cur = PRED_CACHE.get(rel) or {}
    cur.update({k: v for k, v in rec.items() if v is not None})
    if cur:
        PRED_CACHE.set(rel, cur)
    return rec

@app.post("/api/cot/save", response_model=SaveResp)
def save(req: SaveReq):
    t0 = time.time()
    img_path = _safe_join(req.rel)
    if not img_path.is_file():
        raise HTTPException(status_code=404, detail="image not found")

    cache_cur = PRED_CACHE.get(req.rel) or {}
    for k in ("age_years","sex","view"):
        v = getattr(req, k)
        if v is not None:
            cache_cur[k] = v
    if cache_cur:
        PRED_CACHE.set(req.rel, cache_cur)

    fb = {
        "image_path": str(img_path),
        "image_rel": req.rel,
        "report": (req.edited_report or "").strip(),
        "comment": (req.clinician_comment or "").strip(),
        "cot": req.cot or {},
        "age_years": req.age_years,
        "sex": req.sex,
        "view": req.view,
        "ts": time.time(),
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    append_feedback_line(FEEDBACK_JSONL, fb)
    try:
        cot_text_str = ""
        if req.cot:
            cot_text_str = json.dumps(req.cot, ensure_ascii=False)

        csv_row = {
            "case_id": req.rel,
            "train_image_path": str(img_path),
            "user_image_path": "",  
            "cot_text": cot_text_str,
            "lora_report": (req.first_gen_report or "").strip(),
            "trl_report": (req.edited_report or "").strip(),
            "clinical_score": "",
            "expression_score": "",
            "logic_score": "",
            "better_model": "",
            "comment": (req.clinician_comment or "").strip(),
        }
        append_cot_csv_row(csv_row)
    except Exception as e:
        print(f"[WARN] CSV logging failed for {req.rel}: {e}")

    def _cot_to_note(cot: Optional[Dict[str, Optional[str]]]) -> str:
        if not cot: return ""
        parts=[]
        if cot.get("first_look"):      parts.append(f"FirstLook: {cot['first_look']}")
        if cot.get("not_on_image"):    parts.append(f"NotOnImage: {cot['not_on_image']}")
        if cot.get("uncertainty"):     parts.append(f"Uncertainty: {cot['uncertainty']}")
        if cot.get("final_diagnosis"): parts.append(f"FinalDx: {cot['final_diagnosis']}")
        if cot.get("next_step"):       parts.append(f"NextStep: {cot['next_step']}")
        return " | ".join(parts)

    meta = {}
    if req.view: meta["view"] = req.view
    if req.sex: meta["sex"] = req.sex
    if req.age_years is not None: meta["age_years"] = req.age_years

    clinician_note = _cot_to_note(req.cot) or req.clinician_comment
    prompt = build_prompt_existing_with_feedback(meta, clinician_note)

    append_dpo_pair(
        DPO_JSONL,
        prompt=prompt,
        chosen=(req.edited_report or "").strip(),
        rejected=(req.first_gen_report or "").strip(),
        image_path=str(img_path),
    )

    dt = (time.time() - t0) * 1000.0
    print(f"[SAVE] {req.rel} saved in {dt:.1f} ms")
    return SaveResp(ok=True, wrote_feedback=True, wrote_dpo=True)
