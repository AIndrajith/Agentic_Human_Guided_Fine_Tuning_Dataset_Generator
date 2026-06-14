# web_api

FastAPI service + LangGraph QA pipeline. Independent uv project
(own `.venv`, web-only deps — no torch / Marker).

> Run from the **repo root**, not this folder, so `web_api.*` imports resolve.

## Containers needed
MongoDB · Redis · Qdrant · MinIO   (start from root: `docker compose up -d` + mongo)

## Install
```
cd web_api && uv sync
```

## Run  (from repo root)
```
web_api\.venv\Scripts\activate
uvicorn web_api.main:app --reload --port 8000
```
Without activating:
```
web_api\.venv\Scripts\uvicorn.exe web_api.main:app --reload --port 8000
```

## Verify
```
http://localhost:8000        -> {"message": "..."}
http://localhost:8000/docs   -> Swagger
```

## Env (shared root `.env`)
`MONGODB_URL`, `DATABASE_NAME`, `MINIO_*`, `ENCRYPTION_KEY`.
Per-provider LLM keys are **not** env vars — they're set at runtime via the
Credentials API and Fernet-encrypted in Mongo.
