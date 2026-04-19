from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "artifacts" / "phase1_dataset"
OUTPUT_DIR = ROOT / "artifacts" / "baseline_tfidf_logreg"


def combine_text(frame: pd.DataFrame) -> list[str]:
    headline = frame["headline"].fillna("").astype(str).str.strip()
    content = frame["content"].fillna("").astype(str).str.strip()
    category = frame["category"].fillna("").astype(str).str.strip()
    return (
        "[CATEGORY] "
        + category
        + " [HEADLINE] "
        + headline
        + " [CONTENT] "
        + content
    ).tolist()


def load_split(name: str) -> tuple[list[str], list[str]]:
    frame = pd.read_csv(DATA_DIR / f"{name}.csv")
    texts = combine_text(frame)
    labels = frame["label"].astype(str).tolist()
    return texts, labels


def compute_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, object]:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
        "weighted_f1": round(f1_score(y_true, y_pred, average="weighted"), 4),
        "classification_report": classification_report(
            y_true,
            y_pred,
            digits=4,
            output_dict=True,
        ),
    }


def main() -> None:
    x_train, y_train = load_split("train")
    x_valid, y_valid = load_split("valid")
    x_test, y_test = load_split("test")

    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=120000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    solver="liblinear",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    pipeline.fit(x_train, y_train)

    valid_pred = pipeline.predict(x_valid)
    test_pred = pipeline.predict(x_test)

    valid_metrics = compute_metrics(y_valid, valid_pred)
    test_metrics = compute_metrics(y_test, test_pred)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, OUTPUT_DIR / "model.joblib")

    metrics = {
        "model_name": "tfidf_logistic_regression",
        "features": "category + headline + content",
        "vectorizer": {
            "analyzer": "word",
            "ngram_range": [1, 2],
            "min_df": 2,
            "max_features": 120000,
            "sublinear_tf": True,
        },
        "classifier": {
            "type": "LogisticRegression",
            "solver": "liblinear",
            "max_iter": 1000,
            "random_state": 42,
        },
        "validation": valid_metrics,
        "test": test_metrics,
    }
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary_lines = [
        "# TF-IDF + Logistic Regression Baseline",
        "",
        f"- Validation accuracy: `{valid_metrics['accuracy']}`",
        f"- Validation macro F1: `{valid_metrics['macro_f1']}`",
        f"- Test accuracy: `{test_metrics['accuracy']}`",
        f"- Test macro F1: `{test_metrics['macro_f1']}`",
        "",
        "## Notes",
        "",
        "- Input text is built from `category + headline + content`.",
        "- This is the classical baseline for comparison against BanglaBERT later.",
        "- The serialized model is saved as `model.joblib`.",
        "",
    ]
    (OUTPUT_DIR / "README.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print("Saved baseline artifacts to:", OUTPUT_DIR)
    print("Validation accuracy:", valid_metrics["accuracy"])
    print("Validation macro F1:", valid_metrics["macro_f1"])
    print("Test accuracy:", test_metrics["accuracy"])
    print("Test macro F1:", test_metrics["macro_f1"])


if __name__ == "__main__":
    main()
