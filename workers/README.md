# workers

Celery document-processing workers. Independent uv project
(own `.venv`, heavy ML deps: Marker + torch).

> Run from the **repo root**, not this folder, so `workers.*` imports resolve.

## Containers needed
Redis (broker) · Qdrant · MinIO · MongoDB  + `web_api` running
(workers fetch files & post results via web_api `/internal/*` and webhooks).

## Install
```
cd workers && uv sync     # first sync pulls torch + Marker — large, one-time
```

## Run  (from repo root)
```
workers\.venv\Scripts\activate
celery -A workers.celery_app worker --loglevel=info --pool=solo
```
Without activating:
```
workers\.venv\Scripts\celery.exe -A workers.celery_app worker --loglevel=info --pool=solo
```

`--pool=solo` is **required** on Windows.

## Env (shared root `.env`)
`OPENAI_API_KEY`, `GEMINI_API_KEY`, `MONGODB_URL`, `MINIO_*`, `QDRANT_URL`,
`CELERY_BROKER_URL`, `WEB_API_URL`

## Notes
- First academic PDF run is slow — Marker downloads models once, then caches.
