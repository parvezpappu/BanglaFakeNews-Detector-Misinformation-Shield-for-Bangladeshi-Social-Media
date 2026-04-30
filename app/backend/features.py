from __future__ import annotations

import numpy as np


def build_model_text(headline: str, content: str) -> str:
    return f"[HEADLINE] {headline.strip()} [CONTENT] {content.strip()}"


def basic_features(headline: str, content: str) -> np.ndarray:
    headline_words = headline.split()
    content_words = content.split()
    values = np.array(
        [
            len(headline),
            len(content),
            len(headline_words),
            len(content_words),
            sum(ch.isdigit() for ch in headline),
            sum(ch.isdigit() for ch in content),
            sum(ch in "!?।,:;" for ch in headline),
            sum(ch in "!?।,:;" for ch in content),
            sum(ch.isalpha() and ord(ch) < 128 for ch in headline),  # English chars in headline
            sum(ch.isalpha() and ord(ch) < 128 for ch in content),   # English chars in content
        ],
        dtype=np.float32,
    )
    return values


def confidence_features(probabilities: np.ndarray) -> np.ndarray:
    max_prob = np.max(probabilities)
    min_prob = np.min(probabilities)
    margin = max_prob - min_prob
    entropy = -float(np.sum(probabilities * np.log(np.clip(probabilities, 1e-9, 1.0))))
    return np.array([max_prob, margin, entropy], dtype=np.float32)


def build_xgboost_features(
    embedding: np.ndarray,
    bert_probabilities: np.ndarray,
    headline: str,
    content: str,
) -> np.ndarray:
    parts = [
        embedding.astype(np.float32),
        bert_probabilities.astype(np.float32),
        confidence_features(bert_probabilities),
        basic_features(headline, content),
    ]
    return np.concatenate(parts, axis=0).astype(np.float32)