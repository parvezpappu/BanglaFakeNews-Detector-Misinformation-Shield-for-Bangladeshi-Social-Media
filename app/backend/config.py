from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ENSEMBLE_DIR = ROOT / "artifacts" / "banglabert_xgboost_ensemble_v2"
CATEGORY_MODEL_DIR = ROOT / "artifacts" / "category_model"

MODEL_DIR = Path(os.getenv("BANGLABERT_MODEL_DIR", ENSEMBLE_DIR / "banglabert_model"))
XGBOOST_MODEL_PATH = Path(os.getenv("XGBOOST_MODEL_PATH", ENSEMBLE_DIR / "xgboost_model.joblib"))
STACKING_MODEL_PATH = Path(os.getenv("STACKING_MODEL_PATH", ENSEMBLE_DIR / "stacking_model.joblib"))


MODEL_NAME = os.getenv("BANGLABERT_MODEL_NAME", "csebuetnlp/banglabert")
MODEL_SUBFOLDER = os.getenv("BANGLABERT_MODEL_SUBFOLDER", "")
TOKENIZER_MODEL_NAME = os.getenv("TOKENIZER_MODEL_NAME", "csebuetnlp/banglabert")
TOKENIZER_SUBFOLDER = os.getenv("TOKENIZER_SUBFOLDER", "")
ALLOW_PUBLIC_MODEL_FALLBACK = os.getenv("ALLOW_PUBLIC_MODEL_FALLBACK", "false").lower() == "true"
MAX_LENGTH = 512
ENSEMBLE_BANGLABERT_WEIGHT = 0.45
ENSEMBLE_XGBOOST_WEIGHT = 0.55

GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY", "")
GOOGLE_SEARCH_CX = os.getenv("GOOGLE_SEARCH_CX", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
EVIDENCE_SEARCH_PROVIDER = os.getenv("EVIDENCE_SEARCH_PROVIDER", "auto").lower()
EVIDENCE_SEARCH_RESULTS = int(os.getenv("EVIDENCE_SEARCH_RESULTS", "5"))
EVIDENCE_SEARCH_TIMEOUT = float(os.getenv("EVIDENCE_SEARCH_TIMEOUT", "8"))

TRUSTED_EVIDENCE_DOMAINS = [
    "bssnews.net",
    "bdnews24.com",
    "prothomalo.com",
    "thedailystar.net",
    "dhakatribune.com",
    "banglatribune.com",
    "jugantor.com",
    "kalerkantho.com",
    "somoynews.tv",
    "rumorscanner.com",
    "boombd.com",
    "bangladesh.gov.bd",
]

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
