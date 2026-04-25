from __future__ import annotations

from dataclasses import dataclass
import gc
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
    CATEGORY_MODEL_DIR,
    ROOT,
)
from app.backend.artifacts import ensure_model_artifacts
from app.backend.features import build_model_text, build_xgboost_features


LABELS = ["fake", "real"]
REMOTE_MODEL_FILES = ["config.json", "model.safetensors"]


def resolve_model_source() -> str:
    if MODEL_DIR.exists():
        return str(MODEL_DIR)

    if MODEL_SUBFOLDER:
        local_model_dir = Path("/tmp/banglabert_model")
        local_model_dir.mkdir(parents=True, exist_ok=True)
        subfolder = MODEL_SUBFOLDER.strip().strip("/")

        for filename in REMOTE_MODEL_FILES:
            cached_file = hf_hub_download(
                repo_id=MODEL_NAME,
                filename=f"{subfolder}/{filename}",
            )
            shutil.copyfile(cached_file, local_model_dir / filename)

        return str(local_model_dir)

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
    category: str
    probabilities: dict[str, float]
    branch_probabilities: dict[str, dict[str, float]]


class EnsemblePredictor:
    def __init__(self) -> None:
        ensure_model_artifacts()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_source = resolve_model_source()

        self.tokenizer_kwargs = {}
        if TOKENIZER_SUBFOLDER:
            self.tokenizer_kwargs["subfolder"] = TOKENIZER_SUBFOLDER

        if not XGBOOST_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Missing XGBoost model at {XGBOOST_MODEL_PATH}. Copy it from the Colab export first."
            )
        self.xgboost_model = joblib.load(XGBOOST_MODEL_PATH)
        self.ensemble_banglabert_weight, self.ensemble_xgboost_weight = self._ensemble_weights()

    def _ensemble_weights(self) -> tuple[float, float]:
        metrics_path = ROOT / "artifacts" / "banglabert_xgboost_ensemble_v2" / "metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            alpha = metrics.get("ensemble", {}).get("alpha_for_banglabert")
            if alpha is not None:
                alpha = float(alpha)
                return alpha, 1.0 - alpha
        return ENSEMBLE_BANGLABERT_WEIGHT, ENSEMBLE_XGBOOST_WEIGHT

    def _release_memory(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_values = np.exp(shifted)
        return exp_values / exp_values.sum(axis=1, keepdims=True)

    def _encode(self, tokenizer: AutoTokenizer, text: str) -> dict[str, torch.Tensor]:
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        return {key: value.to(self.device) for key, value in encoded.items()}

    def _bert_outputs(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_NAME, **self.tokenizer_kwargs)
        classifier = AutoModelForSequenceClassification.from_pretrained(self.model_source)
        classifier.to(self.device)
        classifier.eval()

        with torch.no_grad():
            encoded = self._encode(tokenizer, text)
            logits = classifier(**encoded).logits
            probabilities = self._softmax(logits.detach().cpu().numpy())[0]
            base_model = getattr(classifier, classifier.base_model_prefix)
            last_hidden = base_model(**encoded).last_hidden_state[:, 0, :]
            embedding = last_hidden.detach().cpu().numpy()[0].astype(np.float32)

        del classifier, tokenizer, encoded, logits, last_hidden
        self._release_memory()
        return probabilities, embedding

    def predict(self, category: str, headline: str, content: str) -> PredictionResult:
        category, category_probs = self._category_outputs(headline, content)
        text = build_model_text(category, headline, content)
    
        bert_probabilities, embedding = self._bert_outputs(text)
    
        xgb_features = build_xgboost_features(
            embedding=embedding,
            bert_probabilities=bert_probabilities,
            category=category,
            headline=headline,
            content=content,
            category_probs=category_probs,
        ).reshape(1, -1)
        xgb_probabilities = self.xgboost_model.predict_proba(xgb_features)[0]
    
        ensemble_probabilities = (
            self.ensemble_banglabert_weight * bert_probabilities
            + self.ensemble_xgboost_weight * xgb_probabilities
        )
    
        best_idx = int(np.argmax(ensemble_probabilities))
        return PredictionResult(
            label=LABELS[best_idx],
            confidence=float(ensemble_probabilities[best_idx]),
            category=category,
            probabilities={label: float(prob) for label, prob in zip(LABELS, ensemble_probabilities)},
            branch_probabilities={
                "banglabert": {label: float(prob) for label, prob in zip(LABELS, bert_probabilities)},
                "xgboost": {label: float(prob) for label, prob in zip(LABELS, xgb_probabilities)},
            },
        )
    


    def _category_outputs(self, headline: str, content: str) -> tuple[str, np.ndarray]:
        text = f"[HEADLINE] {headline.strip()} [CONTENT] {content.strip()}"
        tokenizer = AutoTokenizer.from_pretrained(str(CATEGORY_MODEL_DIR))
        model = AutoModelForSequenceClassification.from_pretrained(str(CATEGORY_MODEL_DIR))
        model.to(self.device)
        model.eval()

        encoded = tokenizer(
            text,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=-1)
        pred = torch.argmax(logits, dim=1).item()
        category = model.config.id2label[pred]
        category_probs = probs.cpu().numpy()[0].astype(np.float32)

        del model, tokenizer, encoded, logits, probs
        self._release_memory()
        return category, category_probs
