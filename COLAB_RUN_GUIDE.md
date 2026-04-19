# Colab Run Guide

Use a `GPU` runtime in Google Colab.

## 1. Upload your project

Put this project in Google Drive or upload a zip to Colab, then make sure these files exist:

- `artifacts/phase1_dataset/train.csv`
- `artifacts/phase1_dataset/valid.csv`
- `artifacts/phase1_dataset/test.csv`
- `scripts/train_banglabert_xgboost_ensemble.py`

## 2. Install dependencies

```python
!pip install -q torch transformers datasets accelerate evaluate pandas scikit-learn xgboost joblib sentencepiece
```

## 3. Check GPU

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
```

## 4. Run the full ensemble training

```python
!python scripts/train_banglabert_xgboost_ensemble.py \
  --epochs 2 \
  --train-batch-size 8 \
  --eval-batch-size 16 \
  --gradient-accumulation-steps 1 \
  --max-length 256 \
  --xgb-estimators 400 \
  --xgb-max-depth 6 \
  --xgb-learning-rate 0.05
```

## 5. Read results

```python
import json
from pathlib import Path

metrics_path = Path("artifacts/banglabert_xgboost_ensemble/metrics.json")
metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
print(json.dumps(metrics, ensure_ascii=False, indent=2))
```

## 6. Download trained artifacts

Important outputs:

- `artifacts/banglabert_xgboost_ensemble/banglabert_model`
- `artifacts/banglabert_xgboost_ensemble/xgboost_model.joblib`
- `artifacts/banglabert_xgboost_ensemble/metrics.json`

## Recommended first run

Start with:

- `epochs=2`
- `max_length=256`
- `train_batch_size=8`
- `eval_batch_size=16`

If Colab GPU memory is enough, then try:

- `epochs=3`
- `max_length=320`
- `train_batch_size=8`

## If Colab runs out of memory

Reduce one or more of:

- `--train-batch-size 4`
- `--eval-batch-size 8`
- `--max-length 192`
- `--gradient-accumulation-steps 2`
