from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from app.backend.config import (
    ALLOW_PUBLIC_MODEL_FALLBACK,
    ENSEMBLE_BANGLABERT_WEIGHT,
    ENSEMBLE_XGBOOST_WEIGHT,
    MODEL_DIR,
    MODEL_NAME,
    MODEL_SUBFOLDER,
    MAX_LENGTH,
    TOKENIZER_MODEL_NAME,
    TOKENIZER_SUBFOLDER,
    XGBOOST_MODEL_PATH,
)
from app.backend.features import build_model_text, build_xgboost_features


LABELS = ["fake", "real"]


def resolve_model_source() -> str:
    if MODEL_DIR.exists():
        return str(MODEL_DIR)

    if MODEL_SUBFOLDER:
        local_repo_dir = Path(
            snapshot_download(
                repo_id=MODEL_NAME,
                allow_patterns=f"{MODEL_SUBFOLDER.rstrip('/')}/*",
            )
        )
        local_model_dir = local_repo_dir / MODEL_SUBFOLDER
        if local_model_dir.exists():
            return str(local_model_dir)
        raise FileNotFoundError(
            f"Downloaded {MODEL_NAME}, but could not find subfolder {MODEL_SUBFOLDER}."
        )

    if ALLOW_PUBLIC_MODEL_FALLBACK:
        return MODEL_NAME

    raise FileNotFoundError(
        f"Missing fine-tuned BanglaBERT model at {MODEL_DIR}. "
        "Add the exported banglabert_model folder or set BANGLABERT_MODEL_DIR."
    )


@dataclass
class PredictionResult:
    label: str
    confidence: float
    probabilities: dict[str, float]
    branch_probabilities: dict[str, dict[str, float]]


class EnsemblePredictor:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_source = resolve_model_source()

        tokenizer_kwargs = {}
        if TOKENIZER_SUBFOLDER:
            tokenizer_kwargs["subfolder"] = TOKENIZER_SUBFOLDER

        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_NAME, **tokenizer_kwargs)
        self.classifier = AutoModelForSequenceClassification.from_pretrained(model_source)
        self.encoder = AutoModel.from_pretrained(model_source)
        self.classifier.to(self.device)
        self.encoder.to(self.device)
        self.classifier.eval()
        self.encoder.eval()

        if not XGBOOST_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Missing XGBoost model at {XGBOOST_MODEL_PATH}. Copy it from the Colab export first."
            )
        self.xgboost_model = joblib.load(XGBOOST_MODEL_PATH)

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_values = np.exp(shifted)
        return exp_values / exp_values.sum(axis=1, keepdims=True)

    def _encode(self, text: str) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        return {key: value.to(self.device) for key, value in encoded.items()}

    def _bert_probabilities(self, text: str) -> np.ndarray:
        encoded = self._encode(text)
        with torch.no_grad():
            logits = self.classifier(**encoded).logits
        return self._softmax(logits.detach().cpu().numpy())[0]

    def _embedding(self, text: str) -> np.ndarray:
        encoded = self._encode(text)
        with torch.no_grad():
            last_hidden = self.encoder(**encoded).last_hidden_state[:, 0, :]
        return last_hidden.detach().cpu().numpy()[0].astype(np.float32)

    def predict(self, category: str, headline: str, content: str) -> PredictionResult:
        text = build_model_text(category, headline, content)

        bert_probabilities = self._bert_probabilities(text)
        embedding = self._embedding(text)

        xgb_features = build_xgboost_features(
            embedding=embedding,
            bert_probabilities=bert_probabilities,
            category=category,
            headline=headline,
            content=content,
        ).reshape(1, -1)
        xgb_probabilities = self.xgboost_model.predict_proba(xgb_features)[0]

        ensemble_probabilities = (
            ENSEMBLE_BANGLABERT_WEIGHT * bert_probabilities
            + ENSEMBLE_XGBOOST_WEIGHT * xgb_probabilities
        )

        best_idx = int(np.argmax(ensemble_probabilities))
        return PredictionResult(
            label=LABELS[best_idx],
            confidence=float(ensemble_probabilities[best_idx]),
            probabilities={label: float(prob) for label, prob in zip(LABELS, ensemble_probabilities)},
            branch_probabilities={
                "banglabert": {label: float(prob) for label, prob in zip(LABELS, bert_probabilities)},
                "xgboost": {label: float(prob) for label, prob in zip(LABELS, xgb_probabilities)},
            },
        )
