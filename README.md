# Luman: Synthetic Data Generation

Luman turns your PDFs into **question–answer datasets for fine-tuning LLMs**. AI writes the question–answer pairs; people check the uncertain ones before they go into the dataset.

> **Status: in development.** Document ingestion works end to end. QA generation and human review are designed but not built yet (see [Roadmap](#roadmap)).

## How it works

```
 1. INGEST                    2. GENERATE                      3. REVIEW & EXPORT
 PDFs ─► extract text ─►      pick a chunk ─► write          score 9–10 ─► accepted automatically
         chunk ─► embed       questions ─► retrieve           score 5–8  ─► a person accepts,
         ─► vector search     context ─► answer ─► score                   rejects, edits or retries
                                                              score < 5  ─► rejected automatically
                                                              ─► export: Chat · CoT · DPO (JSONL)
```

1. **Ingest.** Upload fiction or academic PDFs to a project. A worker extracts the text (academic papers are converted to Markdown, and figures get AI-written descriptions), splits it into chunks, and stores searchable vectors.
2. **Generate.** An AI pipeline writes questions from the documents, answers them using retrieved passages, and scores each answer against those passages.
3. **Review & export.** Only the uncertain pairs (score 5–8) go to a person. Accepted pairs are exported in standard fine-tuning formats. The retry history also provides preference pairs for DPO.

**Roles:** *admins* manage users and API keys. A project *owner* configures the project (documents, which model runs each step, prompts). *Workers* review QA pairs.

## Roadmap

| Area | State |
|---|---|
| Users, invites, roles, projects, members | ✅ Done |
| Document upload, processing jobs, status tracking | ✅ Done |
| Provider credentials + model choice for each step (OpenAI, Anthropic, Google, Cohere, Ollama, ...) | ✅ Done |
| Text extraction, chunking, embeddings, Qdrant storage (workers) | 🟡 Works, known bugs being fixed |
| QA generation pipeline (LangGraph) | ⏳ Planned |
| Human review queue, dataset export | ⏳ Planned |
| Web UI | 🟡 Early |

## Tech stack

**API:** FastAPI · PostgreSQL (SQLAlchemy async, Alembic) · MinIO · LangGraph
**Processing:** Celery + Redis · PyMuPDF · Marker · Chonkie · Qdrant (dense + BM25)
**UI:** React + Vite

## Repository layout

```
web_api/      FastAPI service: auth, projects, documents, credentials, processing
workers/      Celery workers: PDF extraction, chunking, embedding, Qdrant
front-end/    React UI
infra/        Postgres first-start setup (database roles)
docs/         design, development guide, brand assets
```

## Quick start

```powershell
copy .env.example .env                    # fill in the secrets
docker compose up -d                      # postgres, redis, qdrant, minio
cd web_api ; uv sync ; uv run alembic upgrade head ; cd ..
uv run --project web_api python -m web_api.scripts.create_admin --email you@example.com --username admin
uv run --project web_api uvicorn web_api.main:app --reload --loop asyncio:SelectorEventLoop
```

Then open http://localhost:8000/docs. For workers, the front-end, tests and troubleshooting, see the **[development guide](docs/development.md)**.

## Documentation

- [docs/api.md](docs/api.md): API guide with every endpoint, request/response fields, and the processing flow
- [docs/development.md](docs/development.md): setup, running, tests, migrations, troubleshooting
- [docs/design.md](docs/design.md): full architecture, pipelines, data model, API design
- [web_api/README.md](web_api/README.md) · [workers/README.md](workers/README.md): service details
