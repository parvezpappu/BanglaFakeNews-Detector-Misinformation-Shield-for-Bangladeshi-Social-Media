# Backend Inference

This backend serves the final project model as a `BanglaBERT + XGBoost ensemble`.

## What It Needs

Place these exported artifacts from Colab into `Improved/`:

- `xgboost_model.joblib`
- `banglabert_model/`

The local folder should look like:

```text
Improved/
  banglabert_model/
  xgboost_model.joblib
  metrics.json
  README.md
```

## Run

Install dependencies:

```powershell
python -m pip install -r app/backend/requirements.txt
```

Start the API:

```powershell
uvicorn app.backend.main:app --reload
```

## Endpoints

- `GET /health`
- `POST /predict`

Example request body:

```json
{
  "category": "National",
  "headline": "নমুনা শিরোনাম",
  "content": "নমুনা সংবাদ কনটেন্ট",
  "include_evidence": true
}
```

## Optional Evidence Search

The easiest provider is Tavily because it needs only one key:

```text
TAVILY_API_KEY=your_tavily_api_key
```

The backend also has a Google fallback, but Google Custom Search JSON API is closed to many new customers. Use it only if your Google project already has JSON API access:

```text
GOOGLE_SEARCH_API_KEY=your_google_api_key
GOOGLE_SEARCH_CX=your_search_engine_id
```

Without search credentials, prediction still works and returns `evidence.verdict_hint = "model_only"`.

## Railway Deployment

Recommended approach:

- Deploy the backend as a Docker service on Railway
- Use the Dockerfile at `app/backend/Dockerfile`
- Set the service port to `8000`

Important environment variable:

```text
CORS_ALLOW_ORIGINS=https://your-frontend.vercel.app,http://localhost:5173
GOOGLE_SEARCH_API_KEY=your_google_api_key
GOOGLE_SEARCH_CX=your_search_engine_id
```
