from __future__ import annotations

from functools import lru_cache
import os
import traceback

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.backend.evidence import check_evidence
from app.backend.history_store import (
    delete_prediction,
    history_status_message,
    is_history_enabled,
    list_predictions,
    save_prediction,
)
from app.backend.predictor import EnsemblePredictor


class PredictRequest(BaseModel):
    headline: str
    content: str
    include_evidence: bool = Field(default=True)


class EvidenceRequest(BaseModel):
    headline: str
    content: str = ""


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
    probabilities: dict[str, float]
    branch_probabilities: dict[str, dict[str, float]]
    evidence: EvidenceResponse | None = None


class HistoryItemResponse(BaseModel):
    id: str
    headline: str
    content: str
    label: str
    confidence: float
    probabilities: dict[str, float]
    branch_probabilities: dict[str, dict[str, float]]
    created_at: str


class HistoryResponse(BaseModel):
    enabled: bool
    message: str = ""
    items: list[HistoryItemResponse]


class DeleteHistoryResponse(BaseModel):
    deleted: bool


app = FastAPI(title="Bangla Fake News Detector API", version="0.4.0")

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


@app.get("/history", response_model=HistoryResponse)
def history(limit: int = 20) -> HistoryResponse:
    return HistoryResponse(
        enabled=is_history_enabled(),
        message=history_status_message(),
        items=[
            HistoryItemResponse(
                id=item["id"],
                headline=item["headline"],
                content=item["content"],
                label=item["label"],
                confidence=item["confidence"],
                probabilities=item["probabilities"],
                branch_probabilities=item["branch_probabilities"],
                created_at=item["created_at"],
            )
            for item in list_predictions(limit=limit)
        ],
    )


@app.delete("/history/{prediction_id}", response_model=DeleteHistoryResponse)
def delete_history_item(prediction_id: str) -> DeleteHistoryResponse:
    deleted = delete_prediction(prediction_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="History item not found.")
    return DeleteHistoryResponse(deleted=True)


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    try:
        predictor = get_predictor()
        result = predictor.predict(
            headline=payload.headline,
            content=payload.content,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        error_tail = traceback.format_exc().splitlines()[-6:]
        raise HTTPException(
            status_code=500,
            detail={"error": str(exc), "traceback_tail": error_tail},
        ) from exc

    evidence = None
    if payload.include_evidence:
        evidence_result = check_evidence(
            category="",
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
                    title=item.title, link=item.link,
                    snippet=item.snippet, source=item.source,
                )
                for item in evidence_result.items
            ],
            note=evidence_result.note,
        )

    try:
        save_prediction(
            headline=payload.headline,
            content=payload.content,
            label=result.label,
            confidence=result.confidence,
            probabilities=result.probabilities,
            branch_probabilities=result.branch_probabilities,
        )
    except Exception:
        traceback.print_exc()

    return PredictResponse(
        label=result.label,
        confidence=result.confidence,
        probabilities=result.probabilities,
        branch_probabilities=result.branch_probabilities,
        evidence=evidence,
    )


@app.post("/check-evidence")
def check_evidence_endpoint(payload: EvidenceRequest) -> EvidenceResponse:
    from app.backend.evidence_search import search_evidence

    result = search_evidence(
        headline=payload.headline,
        content=payload.content,
    )

    return EvidenceResponse(
        status=result.status,
        verdict_hint="",
        query=result.query,
        search_url=result.search_url,
        items=[
            EvidenceItemResponse(
                title=item.title, link=item.link,
                snippet=item.snippet, source=item.source,
            )
            for item in result.items
        ],
        note=result.note,
    )
