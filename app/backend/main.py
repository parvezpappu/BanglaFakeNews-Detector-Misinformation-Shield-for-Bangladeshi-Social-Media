from __future__ import annotations

from functools import lru_cache
import os
import traceback

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.backend.predictor import EnsemblePredictor


class PredictRequest(BaseModel):
    category: str = Field(default="National")
    headline: str
    content: str


class PredictResponse(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]
    branch_probabilities: dict[str, dict[str, float]]


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


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    try:
        predictor = get_predictor()
        result = predictor.predict(
            category=payload.category,
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

    return PredictResponse(
        label=result.label,
        confidence=result.confidence,
        probabilities=result.probabilities,
        branch_probabilities=result.branch_probabilities,
    )
