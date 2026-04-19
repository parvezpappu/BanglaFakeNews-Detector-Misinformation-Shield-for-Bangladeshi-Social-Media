# Deployment Guide

Recommended hosting split:

- Frontend: `Vercel`
- Backend: `Railway`

This is the safest setup for the current project because the backend includes a large BanglaBERT model.

## 1. Backend on Railway

Deploy the repository to Railway as a Docker service.

### Railway settings

- Root directory: project root
- Dockerfile path: `app/backend/Dockerfile`
- Port: `8000`

### Required backend files

These must exist in the repo before deployment:

- `Improved/xgboost_model.joblib`
- `Improved/banglabert_model-20260418T171553Z-3-001/banglabert_model`

### Railway environment variable

Set:

```text
CORS_ALLOW_ORIGINS=https://your-frontend.vercel.app,http://localhost:5173
```

### Backend health check

Use:

```text
/health
```

## 2. Frontend on Vercel

Create a Vercel project from `app/frontend`.

### Vercel settings

- Framework preset: `Vite`
- Root directory: `app/frontend`

### Frontend environment variable

Set:

```text
VITE_API_BASE_URL=https://your-backend-domain.up.railway.app
```

Then redeploy.

## 3. Deployment order

1. Deploy backend to Railway
2. Copy the Railway public URL
3. Set `VITE_API_BASE_URL` on Vercel
4. Deploy frontend to Vercel
5. Add the Vercel URL to `CORS_ALLOW_ORIGINS` on Railway
6. Redeploy Railway if needed

## 4. Final check

After deployment:

- Open frontend URL
- Submit a Bengali headline and content
- Confirm the result card updates successfully
