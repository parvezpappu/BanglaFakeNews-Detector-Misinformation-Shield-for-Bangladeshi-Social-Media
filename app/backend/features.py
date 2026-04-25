from __future__ import annotations

import numpy as np

from app.backend.config import CATEGORY_VOCAB


def build_model_text(category: str, headline: str, content: str) -> str:
    return f"[CATEGORY] {category.strip()} [HEADLINE] {headline.strip()} [CONTENT] {content.strip()}"


def basic_features(category: str, headline: str, content: str) -> np.ndarray:
    headline_words = headline.split()
    content_words = content.split()
    values = np.array(
        [
            len(headline),
            len(content),
            len(headline_words),
            len(content_words),
            len(category),
            sum(ch.isdigit() for ch in headline),
            sum(ch.isdigit() for ch in content),
            sum(ch in "!?।,:;" for ch in headline),
            sum(ch in "!?।,:;" for ch in content),
        ],
        dtype=np.float32,
    )
    return values


def category_features(category: str) -> np.ndarray:
    normalized = category.strip()
    return np.array([1.0 if normalized == value else 0.0 for value in CATEGORY_VOCAB], dtype=np.float32)


def confidence_features(probabilities: np.ndarray) -> np.ndarray:
    max_prob = np.max(probabilities)
    min_prob = np.min(probabilities)
    margin = max_prob - min_prob
    entropy = -float(np.sum(probabilities * np.log(np.clip(probabilities, 1e-9, 1.0))))
    return np.array([max_prob, margin, entropy], dtype=np.float32)


def build_xgboost_features(
    embedding: np.ndarray,
    bert_probabilities: np.ndarray,
    category: str,
    headline: str,
    content: str,
    category_probs: np.ndarray,
) -> np.ndarray:
    parts = [
        embedding.astype(np.float32),
        bert_probabilities.astype(np.float32),
        confidence_features(bert_probabilities),
        basic_features(category, headline, content),
        category_features(category),
        category_probs.astype(np.float32),
    ]
    return np.concatenate(parts, axis=0).astype(np.float32)
