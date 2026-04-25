from __future__ import annotations

from functools import lru_cache
import json
import os
import traceback

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
from pydantic import BaseModel, Field

from app.backend.artifacts import DEFAULT_HF_MODEL_REPO, REQUIRED_ARTIFACT_FILES, ensure_model_artifacts
from app.backend.config import CATEGORY_MODEL_DIR, MODEL_DIR, ROOT, XGBOOST_MODEL_PATH
from app.backend.evidence import check_evidence
from app.backend.predictor import EnsemblePredictor


class PredictRequest(BaseModel):
    category: str | None = None
    headline: str
    content: str
    include_evidence: bool = Field(default=True)


class EvidenceItemResponse(BaseModel):
    title: str
    link: str
    snippet: str
    source: str


class EvidenceResponse(BaseModel):
    status: str
    verdict_hint: str
    query: str
    search_url: str
    items: list[EvidenceItemResponse]
    note: str


class PredictResponse(BaseModel):
    label: str
    confidence: float
    category: str
    probabilities: dict[str, float]
    branch_probabilities: dict[str, dict[str, float]]
    evidence: EvidenceResponse | None = None


app = FastAPI(title="Bangla Fake News Detector API", version="0.1.0")

cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:5173").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_predictor() -> EnsemblePredictor:
    return EnsemblePredictor()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model-info")
def model_info() -> dict[str, object]:
    ensure_model_artifacts()

    missing_files = [
        relative_path
        for relative_path in REQUIRED_ARTIFACT_FILES
        if not (ROOT / relative_path).exists()
    ]

    metrics_path = ROOT / "artifacts" / "banglabert_xgboost_ensemble_v2" / "metrics.json"
    label_map_path = ROOT / "artifacts" / "category_model" / "label_map.json"

    metrics = {}
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    category_labels = []
    if label_map_path.exists():
        label_map = json.loads(label_map_path.read_text(encoding="utf-8"))
        category_labels = sorted(label_map.get("label2id", {}).keys())

    xgb_feature_count = None
    if XGBOOST_MODEL_PATH.exists():
        xgb_feature_count = int(joblib.load(XGBOOST_MODEL_PATH).n_features_in_)

    return {
        "hf_model_repo": os.getenv("HF_MODEL_REPO", DEFAULT_HF_MODEL_REPO),
        "hf_model_revision": os.getenv("HF_MODEL_REVISION", "main"),
        "hf_token_configured": bool(os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")),
        "model_dir": str(MODEL_DIR),
        "category_model_dir": str(CATEGORY_MODEL_DIR),
        "xgboost_model_path": str(XGBOOST_MODEL_PATH),
        "all_required_files_present": not missing_files,
        "missing_files": missing_files,
        "xgb_feature_count": xgb_feature_count,
        "using_v2_xgboost_features": xgb_feature_count == 806,
        "category_count": len(category_labels),
        "category_labels": category_labels,
        "metrics": {
            "xgb_feature_count": metrics.get("xgb_feature_count"),
            "category_model_used": metrics.get("category_model_used"),
            "alpha_for_banglabert": metrics.get("ensemble", {}).get("alpha_for_banglabert"),
            "alpha_for_xgboost": metrics.get("ensemble", {}).get("alpha_for_xgboost"),
            "ensemble_test_macro_f1": metrics.get("ensemble", {}).get("test", {}).get("macro_f1"),
        },
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    try:
        predictor = get_predictor()
        category = payload.category or "National"  # default fallback

        result = predictor.predict(
            category=category,
            headline=payload.headline,
            content=payload.content,
        )

    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        error_tail = traceback.format_exc().splitlines()[-6:]
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(exc),
                "traceback_tail": error_tail,
            },
        ) from exc

    evidence = None
    if payload.include_evidence:
        evidence_result = check_evidence(
            category=payload.category,
            headline=payload.headline,
            content=payload.content,
            model_label=result.label,
        )
        evidence = EvidenceResponse(
            status=evidence_result.status,
            verdict_hint=evidence_result.verdict_hint,
            query=evidence_result.query,
            search_url=evidence_result.search_url,
            items=[
                EvidenceItemResponse(
                    title=item.title,
                    link=item.link,
                    snippet=item.snippet,
                    source=item.source,
                )
                for item in evidence_result.items
            ],
            note=evidence_result.note,
        )

    return PredictResponse(
        label=result.label,
        confidence=result.confidence,
        category=result.category,  
        probabilities=result.probabilities,
        branch_probabilities=result.branch_probabilities,
        evidence=evidence,
    )
