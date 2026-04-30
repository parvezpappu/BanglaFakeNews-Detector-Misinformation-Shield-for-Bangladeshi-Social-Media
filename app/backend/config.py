from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

def load_local_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


load_local_env(ROOT / ".env")

ENSEMBLE_DIR = ROOT / "artifacts" / "banglabert_lightgbm_ensemble"

MODEL_DIR = Path(os.getenv("BANGLABERT_MODEL_DIR", ENSEMBLE_DIR / "banglabert_model"))
LGBM_MODEL_PATH = Path(os.getenv("LGBM_MODEL_PATH", ENSEMBLE_DIR / "lightgbm_model.joblib"))
STACKING_MODEL_PATH = Path(os.getenv("STACKING_MODEL_PATH", ENSEMBLE_DIR / "stacking_model.joblib"))

MODEL_NAME = os.getenv("BANGLABERT_MODEL_NAME", "csebuetnlp/banglabert")
MODEL_SUBFOLDER = os.getenv("BANGLABERT_MODEL_SUBFOLDER", "")
TOKENIZER_MODEL_NAME = os.getenv("TOKENIZER_MODEL_NAME", "csebuetnlp/banglabert")
TOKENIZER_SUBFOLDER = os.getenv("TOKENIZER_SUBFOLDER", "")
ALLOW_PUBLIC_MODEL_FALLBACK = os.getenv("ALLOW_PUBLIC_MODEL_FALLBACK", "false").lower() == "true"
MAX_LENGTH = 512
ENSEMBLE_BANGLABERT_WEIGHT = 0.50
ENSEMBLE_LIGHTGBM_WEIGHT = 0.50

# Evidence Search Settings
EVIDENCE_SEARCH_PROVIDER = os.getenv("EVIDENCE_SEARCH_PROVIDER", "duckduckgo").lower()
EVIDENCE_SEARCH_RESULTS = int(os.getenv("EVIDENCE_SEARCH_RESULTS", "5"))
EVIDENCE_SEARCH_TIMEOUT = float(os.getenv("EVIDENCE_SEARCH_TIMEOUT", "10"))

# API Keys 
GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY", "")
GOOGLE_SEARCH_CX = os.getenv("GOOGLE_SEARCH_CX", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", os.getenv("SERPAPI_KEY", ""))

# MongoDB History Settings
MONGODB_URI = os.getenv("MONGODB_URI", "")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "bangla_fake_news")
MONGODB_HISTORY_COLLECTION = os.getenv("MONGODB_HISTORY_COLLECTION", "prediction_history")

# Trusted Bangladeshi News Domains
TRUSTED_EVIDENCE_DOMAINS = [
    "prothomalo.com",
    "bdnews24.com",
    "thedailystar.net",
    "dhakatribune.com",
    "banglatribune.com",
    "jugantor.com",
    "kalerkantho.com",
    "somoynews.tv",
    "bssnews.net",
    "bangladesh.gov.bd",
    "rumorscanner.com",
    "boombd.com",
]
