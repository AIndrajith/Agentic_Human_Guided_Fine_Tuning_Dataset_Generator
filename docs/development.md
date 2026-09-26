# Development Guide

How to run the project locally. `web_api/` and `workers/` are **separate uv projects** (the workers carry heavy ML dependencies the API doesn't need). Run everything from the **repo root** so `web_api.*` / `workers.*` imports resolve.

## Prerequisites
- Docker Desktop
- [uv](https://docs.astral.sh/uv/) (Python 3.11+)
- Node 20+ (front-end only)

## 1. First-time setup

```powershell
# 1. Config: copy the template and fill in the secrets (generation commands are in the file)
copy .env.example .env

# 2. Install dependencies (the first workers sync pulls torch + Marker: large, one-time)
cd web_api ; uv sync ; cd ..
cd workers ; uv sync ; cd ..

# 3. Start infrastructure: postgres, redis, qdrant, minio, flower
docker compose up -d
#    Postgres roles (synth_owner / synth_app) are created on the FIRST start only,
#    from infra/postgres/init/01-roles.sh, using the passwords in .env.

# 4. Database schema, LangGraph checkpoint tables, first admin
cd web_api ; uv run alembic upgrade head ; uv run alembic -x db=test upgrade head ; cd ..
uv run --project web_api python -m web_api.scripts.setup_checkpointer
uv run --project web_api python -m web_api.scripts.create_admin --email you@example.com --username admin
```

## 2. Run

```powershell
docker compose start      # if the containers are stopped

# Terminal 1: API  -> http://localhost:8000/docs
uv run --project web_api uvicorn web_api.main:app --reload --port 8000 --loop asyncio:SelectorEventLoop

# Terminal 2: document-processing worker
uv run --project workers celery -A workers.celery_app worker --loglevel=info --pool=solo

# Terminal 3: front-end
cd front-end ; npm install ; npm run dev
```

On Windows, `--loop asyncio:SelectorEventLoop` (psycopg async can't use the Proactor loop) and `--pool=solo` (Celery) are both required.

| Service | URL |
|---|---|
| API docs | http://localhost:8000/docs |
| Qdrant | http://localhost:6333/dashboard |
| MinIO console | http://localhost:9001 |
| Flower (Celery) | http://localhost:5555 |
| Postgres shell | `docker exec -it synthetic_data_postgres psql -U postgres -d synthetic_data` |

## 3. Everyday tasks

```powershell
# Tests (synthetic_data_test database; email, MinIO, Celery, Qdrant and model calls are faked)
uv run --project web_api pytest web_api/tests
# Worker tests (no network: BM25, point ids, retry rules)
uv run --project workers pytest workers/tests

# After changing web_api/db/models: create a migration, REVIEW the file, then apply it
cd web_api
uv run alembic revision --autogenerate -m "describe the change"
uv run alembic upgrade head
uv run alembic -x db=test upgrade head
cd ..

# Stop infrastructure (keeps data)  /  wipe all volumes (dev only)
docker compose stop
docker compose down -v
```

## 4. Configuration

All settings live in the repo-root `.env`, which both services read. `.env.example` lists every key.

- **Database URLs:** use `127.0.0.1`, not `localhost`. On Windows, `localhost` tries IPv6 first, which adds about 2 minutes to every connection.
- **Two database roles:** `DATABASE_URL` uses `synth_app`, which can only read and write data. `MIGRATION_DATABASE_URL` uses `synth_owner`, for Alembic and checkpointer setup.
- **`INTERNAL_API_TOKEN`:** must be the same for the API and the workers.
- **Email:** if `RESEND_API_KEY` or `FROM_EMAIL` is unset, invite links are written to the API log instead of being emailed.
- **LLM provider keys** are not environment variables. Admins add them in the app (Credentials); the worker fetches them per job from web_api. Only Marker (academic PDFs) still reads `GEMINI_API_KEY` from `.env`.
- **LiteLLM** (both services) downloads its newest model catalog at startup. Set `LITELLM_LOCAL_MODEL_COST_MAP=True` to use the copy bundled with the package (offline; tests do this). The version is pinned in both `pyproject.toml` files; upgrade it on purpose.
- **Qdrant:** one collection per project (`project_<id>`), created by the worker. `QDRANT_URL` is used by the worker and by web_api (to delete collections). The image is pinned to `v1.18.0`: **1.15.2 or newer is required**, because Qdrant computes the BM25 keyword vectors itself. The worker refuses to create collections on an older server.

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| `Psycopg cannot use the 'ProactorEventLoop'` | Start uvicorn with `--loop asyncio:SelectorEventLoop` |
| Every DB call takes ~2 minutes | Use `127.0.0.1` instead of `localhost` in the DB URLs |
| Postgres password authentication failed | `.env` passwords changed after the first start. Run `ALTER ROLE ... PASSWORD`, or (dev only) `docker compose down -v` to re-run init |
| API startup waits ~25 s | MinIO isn't running: `docker compose up -d` |
| Celery hangs on tasks | Missing `--pool=solo` (Windows) |
| `ModuleNotFoundError: web_api` | Run from the repo root |
| Worker: `Qdrant 1.x is too old` | Update the image: `docker compose pull qdrant; docker compose up -d qdrant`. If the old data won't load, remove the `qdrant_data` volume (dev only; the vectors get rebuilt by reprocessing) |
| Worker calls get 401 | `INTERNAL_API_TOKEN` differs between the API and the worker environment |
| First academic PDF is very slow | Marker downloads its models once, then caches them |
