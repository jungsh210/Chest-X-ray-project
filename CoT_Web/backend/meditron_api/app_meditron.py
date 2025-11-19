from __future__ import annotations
import json

from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware

from .Meditron import Meditron  

app = FastAPI()

origins = [
    "http://localhost:3000", 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,        
    allow_credentials=True,
    allow_methods=["*"],          
    allow_headers=["*"],
)

@app.post("/api/meditron/analyze")
def api_meditron_analyze(payload: dict = Body(...)):
    report_text = (
        payload.get("report")
        or payload.get("lingshu_report_text")
        or ""
    ).strip()

    original_report = (
        payload.get("original_report")
        or payload.get("original_report_text")
        or ""
    ).strip()

    demographics_payload = payload.get("demographics") or {}
    age = payload.get("age", demographics_payload.get("age_years"))
    sex = payload.get("sex", demographics_payload.get("sex"))
    view = demographics_payload.get("view")  

    demographics = {}
    if age is not None:
        demographics["age_years"] = age
    if sex:
        demographics["sex"] = sex
    if view:
        demographics["view"] = view

    if not report_text:
        return {
            "differential": [],
            "next_tests": [],
            "notes": "empty report text",
        }

    result_json = Meditron(
        lingshu_report_text=report_text,
        original_report_text=original_report,
        demographics=demographics or None,
    )

    try:
        data = json.loads(result_json)
        if not isinstance(data, dict):
            data = {}
    except Exception:
        data = {}

    differential = data.get("differential") or []
    next_tests = data.get("next_tests") or []
    notes = data.get("notes") or ""

    return {
        "differential": differential,
        "next_tests": next_tests,
        "notes": notes,
    }