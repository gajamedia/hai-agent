# hAI Agent Frontend Starter

React + Vite + Tailwind CSS v3 + React Router + Framer Motion. This MVP calls the existing FastAPI service.

## Install and run

From this `frontend` folder:

```powershell
npm install
Copy-Item .env.example .env
npm run dev
```

Open `http://127.0.0.1:5173`.

Start backend from project root (with venv active):

```powershell
uvicorn api.main:app --env-file .env --reload --log-level debug
```

API base defaults to `http://127.0.0.1:8000`. Override `VITE_API_BASE_URL` in `frontend/.env` and restart Vite if needed.

## CORS (if the browser reports a CORS error)

In `api/main.py`, add these imports:

```python
import os
from fastapi.middleware.cors import CORSMiddleware
```

After creating `application = FastAPI(...)` and before including the router, add:

```python
allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://127.0.0.1:5173,http://localhost:5173",
    ).split(",")
    if origin.strip()
]
application.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)
```

Restart Uvicorn after changing backend CORS configuration. Avoid `allow_origins=["*"]` for an app that may later use credentials.

## Important contract note

`POST /api/v1/cognitive/run` runs a cognitive pass and returns a plan; it does not execute the actions in that plan. The sample request includes episode and deterministic reflection inputs because it enables LLM metacognition.
