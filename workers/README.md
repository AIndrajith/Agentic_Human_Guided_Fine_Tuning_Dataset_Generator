# workers

Celery document-processing workers. Independent uv project
(own `.venv`, heavy ML deps: Marker + torch).

> Run from the **repo root**, not this folder, so `workers.*` imports resolve.

## Containers needed
Redis (broker) · Qdrant  + `web_api` running
(workers never touch Postgres/MinIO directly: they fetch files & post results via
web_api `/internal/*` and webhooks, authenticated with `X-Internal-Token`).

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
`INTERNAL_API_TOKEN` (must match web_api), `WEB_API_URL`, `QDRANT_URL`, `CELERY_BROKER_URL`,
plus `GEMINI_API_KEY` for Marker's own LLM step (academic PDFs).
Loaded automatically from the repo-root `.env`.

Models and provider keys are **not** in `.env`: once per task the worker fetches them from
`GET /internal/projects/{id}/processing-config` and calls every provider through LiteLLM.

## Reliability
- **Each document succeeds or fails on its own** and reports it to web_api; a failed document never
  makes the others run again.
- **Only temporary errors are retried** (network, rate limits, 5xx), at the step that hit them, 3 tries
  with 2 s / 4 s waits (`workers/utils/retry.py`). A wrong key, unknown model or broken PDF fails at once.
- **Reprocessing is safe:** Qdrant point ids come from `document_id + chunk number`, and a document's old
  points are deleted before its new ones are stored.
- **Keyword search (BM25)** is computed by Qdrant itself (built-in `qdrant/bm25`, needs Qdrant 1.15.2+):
  the worker sends the chunk text, Qdrant does tokenizing, stopwords, stemming and stable word ids.
  Search sends the query text the same way: `BM25Service.query_document()`.

## Tests
```
uv run --project workers pytest workers/tests      # from the repo root
```

## Notes
- First academic PDF run is slow — Marker downloads models once, then caches.
