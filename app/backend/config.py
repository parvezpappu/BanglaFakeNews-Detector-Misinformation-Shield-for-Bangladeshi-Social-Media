from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
IMPROVED_DIR = ROOT / "Improved"
MODEL_DIR = IMPROVED_DIR / "banglabert_model"
if not MODEL_DIR.exists():
    nested_model_dir = IMPROVED_DIR / "banglabert_model-20260418T171553Z-3-001" / "banglabert_model"
    if nested_model_dir.exists():
        MODEL_DIR = nested_model_dir
XGBOOST_MODEL_PATH = IMPROVED_DIR / "xgboost_model.joblib"

MODEL_NAME = "csebuetnlp/banglabert"
MAX_LENGTH = 256
ENSEMBLE_BANGLABERT_WEIGHT = 0.45
ENSEMBLE_XGBOOST_WEIGHT = 0.55

# Category order must match the training-time one-hot encoding columns.
CATEGORY_VOCAB = [
    "Crime",
    "Editorial",
    "Education",
    "Entertainment",
    "Finance",
    "International",
    "Lifestyle",
    "Miscellaneous",
    "National",
    "Politics",
    "Sports",
    "Technology",
]
