from __future__ import annotations

import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

from app.backend.config import ROOT


DEFAULT_HF_MODEL_REPO = "ParvezPappu627/bangla-fake-news-banglabert"

REQUIRED_ARTIFACT_FILES = [
    "artifacts/category_model/config.json",
    "artifacts/category_model/label_map.json",
    "artifacts/category_model/model.safetensors",
    "artifacts/category_model/tokenizer.json",
    "artifacts/category_model/tokenizer_config.json",
    "artifacts/category_model/training_args.bin",
    "artifacts/banglabert_xgboost_ensemble_v2/xgboost_model.joblib",
    "artifacts/banglabert_xgboost_ensemble_v2/stacking_model.joblib",
    "artifacts/banglabert_xgboost_ensemble_v2/metrics.json",
    "artifacts/banglabert_xgboost_ensemble_v2/README.md",
    "artifacts/banglabert_xgboost_ensemble_v2/banglabert_model/config.json",
    "artifacts/banglabert_xgboost_ensemble_v2/banglabert_model/model.safetensors",
    "artifacts/banglabert_xgboost_ensemble_v2/banglabert_model/tokenizer.json",
    "artifacts/banglabert_xgboost_ensemble_v2/banglabert_model/tokenizer_config.json",
    "artifacts/banglabert_xgboost_ensemble_v2/banglabert_model/training_args.bin",
]


def ensure_model_artifacts() -> None:
    missing_files = [relative for relative in REQUIRED_ARTIFACT_FILES if not (ROOT / relative).exists()]
    if not missing_files:
        return

    repo_id = os.getenv("HF_MODEL_REPO", DEFAULT_HF_MODEL_REPO)
    revision = os.getenv("HF_MODEL_REVISION") or None
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or None

    for relative_path in missing_files:
        cached_path = hf_hub_download(
            repo_id=repo_id,
            filename=relative_path,
            revision=revision,
            token=token,
        )
        target_path = ROOT / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached_path, target_path)
