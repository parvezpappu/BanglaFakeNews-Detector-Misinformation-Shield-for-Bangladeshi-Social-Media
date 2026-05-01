# Deployment Guide

Recommended hosting split:

- Frontend: `Netlify`
- Backend: `Hugging Face Spaces`

This setup is suitable for the current project because the backend includes a large BanglaBERT model and a LightGBM ensemble branch. Netlify should host only the Vite frontend; `/predict` must point to a separate backend URL.

## 1. Backend on Hugging Face Spaces

Create a Hugging Face Space and select `Docker` as the SDK.

### Required inference files

If you upload artifacts directly into the Space repo, these files must exist:

- `Dockerfile`
- `README.md`
- `app/backend`
- `artifacts/banglabert_lightgbm_ensemble/lightgbm_model.joblib`
- `artifacts/banglabert_lightgbm_ensemble/banglabert_model/config.json`
- `artifacts/banglabert_lightgbm_ensemble/banglabert_model/model.safetensors`
- `artifacts/banglabert_lightgbm_ensemble/banglabert_model/tokenizer.json`
- `artifacts/banglabert_lightgbm_ensemble/banglabert_model/tokenizer_config.json`

If your Space is built from GitHub and `artifacts/` is not committed, upload the
same exported files to a Hugging Face model repo instead and set the environment
variables below. This avoids the production error:

```text
Missing LightGBM model at /app/artifacts/banglabert_lightgbm_ensemble/lightgbm_model.joblib
```

Do not upload:

- `Dataset/`
- `artifacts/banglabert_lightgbm_ensemble/checkpoints/`
- `.env`
- `venv/`

### Hugging Face secrets

In the Space settings, add secrets:

```text
SERPAPI_API_KEY=your_serpapi_key
MONGODB_URI=your_mongodb_connection_string
MONGODB_DATABASE=bangla_fake_news
CORS_ALLOW_ORIGINS=https://your-netlify-site.netlify.app,http://localhost:5173,http://127.0.0.1:5173
BANGLABERT_MODEL_NAME=your-username/your-model-repo
BANGLABERT_MODEL_SUBFOLDER=banglabert_lightgbm_ensemble/banglabert_model
TOKENIZER_MODEL_NAME=your-username/your-model-repo
TOKENIZER_SUBFOLDER=banglabert_lightgbm_ensemble/banglabert_model
LGBM_MODEL_REPO_ID=your-username/your-model-repo
LGBM_MODEL_SUBFOLDER=banglabert_lightgbm_ensemble
LGBM_MODEL_FILENAME=lightgbm_model.joblib
```

### Backend health check

Use:

```text
/health
```

The backend URL will be:

```text
https://your-space-name.hf.space
```

## 2. Frontend on Netlify

Create a Netlify project from `app/frontend`.

### Netlify settings

- Framework preset: `Vite`
- Root directory: `app/frontend`
- Build command: `npm run build`
- Publish directory: `app/frontend/dist`

### Frontend environment variable

Set:

```text
VITE_API_BASE_URL=https://your-space-name.hf.space
```

Then redeploy.

## 3. Deployment order

1. Push backend and model files to Hugging Face Spaces, or upload model files to a Hugging Face model repo and set the model env vars
2. Wait for the Space build to finish
3. Open `https://your-space-name.hf.space/health`
4. Set `VITE_API_BASE_URL` on Netlify
5. Add the Netlify URL to `CORS_ALLOW_ORIGINS` in Hugging Face Space secrets
6. Restart the Space

## 4. Final check

After deployment:

- Open frontend URL
- Submit a Bengali headline and content
- Confirm the result card updates successfully
