# Synthetic Data Generation Platform

A **human-guided QA dataset factory** for LLM fine-tuning. It turns raw PDFs into clean, exportable question–answer datasets (Chat / DPO / CoT) — automating generation while keeping a human in control of every pair that enters the dataset.

> ⚠️ **Work in progress.** This README describes the intended flow. See [PROJECT_PROPOSAL.md](PROJECT_PROPOSAL.md) for the full design.

---

## The Problem

Fine-tuning an LLM on a specific domain needs lots of high-quality QA pairs from your source documents. Writing them by hand doesn't scale; fully automating them produces noisy, unreliable data. This platform sits in between: **the agent generates, the human validates.**

## Core Principle

- **The agent is a tool, not the decision-maker.** Ground truth is the *skill config* defined by the project owner. The quality gate is the *worker*.
- **Owner** owns *how the agent behaves* (prompts, question types, N-per-chunk) — never approves individual pairs.
- **Worker** owns *what happens to the output* (accept / reject / regenerate / edit) — never touches agent config.

---

## The Flow

### Phase 1 — Document Processing (Celery)

Heavy, blocking work. Distributed across Celery workers.

```
PDF upload
  ├─ FICTION  → PyMuPDF text extraction
  └─ ACADEMIC → Marker CLI → Markdown + Gemini vision image captioning
        ↓
Hierarchical chunking (Chonkie)
  ├─ Parent: 30k tokens  → context window
  └─ Child:  800 tokens  → retrieval unit
        ↓
Contextualization — each child chunk gets a 50–200 token
context blurb generated from its parent, then embedded
        ↓
Embeddings + BM25 sparse vectors
        ↓
Qdrant (dense + sparse, filtered by project_id)
```

### Phase 2 — QA Generation (LangGraph + Human Review)

Many small async LLM calls with a human in the loop. **Worker-driven, FIFO pull.**

```
Worker requests work → system atomically assigns next N chunks (Redis LPOP)
        ↓
Per chunk: question_gen (plain service) → N questions
        ↓
Per question: spawn a pair-thread (background async task)
        answer_gen  — hybrid search Qdrant + rerank → grounded answer
        validate    — score 1–10 against the retrieved chunks
        route:
          ≥ 9   → AUTO_ACCEPTED   (terminal)
          < 5   → AUTO_REJECTED   (terminal)
          5–8   → interrupt() → PENDING_HUMAN_REVIEW
        ↓
Worker reviews ONLY the 5–8 band and takes one action:
  • Accept              → ACCEPTED → dataset
  • Reject              → REJECTED (reason logged)
  • Regenerate + comment→ feedback injected, agent retries (max 3)
  • Edit directly       → human-corrected, flagged in audit trail
```

---

## Key Design Decisions

- **Routing is automatic; humans see only the uncertain middle.** ≥9 and <5 resolve without a human; 5–8 pairs pause via LangGraph `interrupt()` and surface in the worker's review queue.
- **Validator scores against grounding.** The retrieved chunks used to write the answer are passed to the validator, so its score has a factual basis.
- **Thread-per-QA-pair.** Each pair is its own short-lived LangGraph thread (`thread_id`). This isolates concurrency (N workers × M pairs = independent checkpoints, no contention), makes auto-routing fall out naturally, and keeps the retry loop inside one thread. The compiled graph is a process-wide singleton; only the `answer_gen → validate → review` portion lives in the graph — chunking and question generation are plain services.
- **FIFO, no reservations.** Chunks are either `PENDING` (in queue) or `ASSIGNED`. No browsing, no cherry-picking, no "abandoned" state. Atomic pops guarantee two workers never get the same chunk.
- **Redis = task queue; PostgreSQL = canonical state + LangGraph checkpoints; MinIO = files and extracted text.** Jobs are pausable/resumable and survive restarts.
- **Provider-agnostic.** Every LLM / embedding / rerank call routes through one factory, configured per-project per-stage. No hardcoded keys — credentials are Fernet-encrypted at rest.

---

## State Machines

```
Chunk:  PENDING → ASSIGNED → COMPLETED

Pair:   GENERATED → ACCEPTED
                  → REJECTED
                  → PENDING_REGENERATION → GENERATED   (retry, max 3)
                  → EDITED → ACCEPTED
```

## Audit Trail

Every pair carries full provenance: source chunk & document, assigned worker, skill config version, validator score + reasoning, worker action, comments, full retry history, human-edited flag, and timestamps on every transition. This is the foundation for future data-cleansing features (filter by worker approval rate, score distribution, question-type balance, etc.).

---

## Tech Stack

| Layer | Technology |
|---|---|
| Web API | FastAPI + asyncio |
| Document processing | Celery + Redis broker |
| QA orchestration | LangGraph (checkpointed to PostgreSQL) |
| Vector DB | Qdrant (dense + sparse / BM25) |
| Database | PostgreSQL 17 (SQLAlchemy 2 async + psycopg 3, Alembic migrations) |
| Object storage | MinIO (uploads, extracted text, images) |
| Extraction | PyMuPDF (fiction), Marker CLI + Gemini (academic) |
| Chunking | Chonkie |
| LLM / Embeddings / Rerank | Configurable per project (OpenAI, Anthropic, Google, Cohere, and more) |
| Auth | JWT (python-jose) + Argon2 password hashing |
| Credential encryption | cryptography (Fernet / AES-128) |

## Export Formats

- **Chat** — standard instruction tuning (`messages` JSONL)
- **CoT** — chain-of-thought with `<think>` blocks
- **DPO** — preference pairs mined from the retry audit trail (early low-scored answer = rejected, final = chosen)

---

*See [PROJECT_PROPOSAL.md](PROJECT_PROPOSAL.md) for the complete architecture, data models, and API design.*
