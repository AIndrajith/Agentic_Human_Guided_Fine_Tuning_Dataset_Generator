# web_api

FastAPI service + LangGraph QA pipeline. Independent uv project
(own `.venv`, web-only deps — no torch / Marker).

> Run from the **repo root**, not this folder, so `web_api.*` imports resolve.

## Containers needed
PostgreSQL · Redis · Qdrant · MinIO   (from root: `docker compose up -d`)

## Install + database
```
cd web_api && uv sync
uv run alembic upgrade head                 # schema (as synth_owner)
cd ..
uv run --project web_api python -m web_api.scripts.setup_checkpointer
uv run --project web_api python -m web_api.scripts.create_admin --email you@x.com --username admin
```

## Run  (from repo root)
```
uv run --project web_api uvicorn web_api.main:app --reload --port 8000 --loop asyncio:SelectorEventLoop
```
`--loop asyncio:SelectorEventLoop` is required on Windows (psycopg async can't use the Proactor loop).

## Test
```
uv run --project web_api pytest web_api/tests      # uses synthetic_data_test; email + MinIO faked
```

## Layout
- `db/models/` — SQLAlchemy tables; `alembic/versions/` — migrations (autogenerate, then review)
- `data_models/` — Pydantic request/response schemas
- `services/` — business logic (take an `AsyncSession`); `routers/` — HTTP layer
- `deps/` — auth (`CurrentUser`, `AdminUser`), project access, internal-token guard

## Env (shared root `.env`, see `.env.example`)
`DATABASE_URL` (synth_app), `MIGRATION_DATABASE_URL` (synth_owner), `JWT_SECRET_KEY`,
`ENCRYPTION_KEY`, `INTERNAL_API_TOKEN`, `MINIO_*`, `RESEND_API_KEY` + `FROM_EMAIL`.
Per-provider LLM keys are **not** env vars — admins add them via the Credentials API
(Fernet-encrypted in Postgres) and projects attach them per stage.

Worker-only routes (`/internal/*`, `/webhooks/*`) require the `X-Internal-Token` header.
