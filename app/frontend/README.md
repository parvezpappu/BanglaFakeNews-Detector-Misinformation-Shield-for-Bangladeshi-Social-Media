# Frontend Demo

This React app is the live demo UI for the Bangla fake news detector.

## Run

Install dependencies:

```powershell
cd app/frontend
npm install
```

Create `.env.local` for production-style API calls:

```powershell
echo VITE_API_BASE_URL=http://127.0.0.1:8000 > .env.local
```

Start the backend in another terminal:

```powershell
set PYTHONPATH=g:\Coarse\Spring-25-26\NLP\Bangla Fake News Detection
uvicorn app.backend.main:app --reload
```

Start the frontend:

```powershell
cd app/frontend
npm run dev
```

Vite proxies `/predict` and `/health` to `http://127.0.0.1:8000`.
