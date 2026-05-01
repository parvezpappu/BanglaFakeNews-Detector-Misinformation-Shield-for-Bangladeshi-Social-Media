---
title: Bangla Fake News Detector API
sdk: docker
app_port: 8000
pinned: false
---

# Bangla Fake News Detector API

FastAPI backend for the BanglaBERT + LightGBM fake news detector.

## Runtime

The Docker image starts:

```bash
python -m uvicorn app.backend.main:app --host 0.0.0.0 --port ${PORT:-8000}
```

## Health Check

```text
/health
```
