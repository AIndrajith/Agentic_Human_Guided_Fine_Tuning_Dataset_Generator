# Synthetic Data Generation Platform
### A Human-Guided QA Dataset Factory for LLM Fine-Tuning

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Phase 1 — Document Processing Pipeline](#3-phase-1--document-processing-pipeline)
4. [Phase 2 — QA Generation Pipeline](#4-phase-2--qa-generation-pipeline)
5. [Human-Guided Workflow](#5-human-guided-workflow)
6. [Role & Permission System](#6-role--permission-system)
7. [Data Models](#7-data-models)
8. [API Design](#8-api-design)
9. [Infrastructure & Tech Stack](#9-infrastructure--tech-stack)
10. [Export Formats](#10-export-formats)
11. [Open Decisions](#11-open-decisions)

---

## 1. Project Overview

### 1.1 The Problem

Fine-tuning a large language model requires high-quality question-answer pairs that reflect the exact knowledge domain of your source documents. Creating these pairs manually is prohibitively slow at scale. Fully automating it without human oversight produces noisy, unreliable data — garbage in, garbage out.

### 1.2 The Solution

This platform automates the QA pair generation process while keeping the human in control of quality. It ingests raw documents (PDFs), processes them into a searchable vector database, then runs an agentic pipeline to generate questions, retrieve grounded answers, and validate quality — routing uncertain pairs to human reviewers before they enter the final dataset.

The result is a clean, exportable fine-tuning dataset in standard formats (Chat, DPO, CoT) that directly reflects the content and quality standards of the source documents.

### 1.3 Key Principles

- **Human-guided, not human-replaced** — humans define how agents behave, set quality thresholds, and review borderline cases
- **Domain-aware** — separate processing strategies for fiction and academic documents
- **Scalable by design** — heavy processing is distributed (Celery); agentic orchestration is async (LangGraph)
- **Resumable** — all long-running jobs support pause and resume via checkpointing
- **Auditable** — every QA pair traces back to its source chunk, retrieval context, validation score, and reviewer

---

## 2. System Architecture

### 2.1 High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT / UI                              │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP
┌──────────────────────────▼──────────────────────────────────────┐
│                      FastAPI  web_api                           │
│                                                                  │
│  Routers: Auth, Files, Projects, Processing,                    │
│           QA Jobs, Skills, Human Review,                        │
│           Dataset Export, Webhooks, Internal                    │
│                                                                  │
│  LangGraph QA Pipeline (async background tasks)                 │
└────────┬──────────────────────────────────────┬─────────────────┘
         │ Celery tasks                          │ Async graph
         ▼                                       ▼
┌────────────────────┐                ┌──────────────────────────┐
│   Celery Workers   │                │   LangGraph QA Graph     │
│                    │                │                          │
│  Document ingest   │                │  chunk → question_gen    │
│  PDF extraction    │                │  → answer_gen → validate │
│  Chunking          │                │  → route → human/accept/ │
│  Embedding         │                │  reject                  │
│  Qdrant storage    │                │                          │
└────────┬───────────┘                └──────────┬───────────────┘
         │                                       │
         ▼                                       ▼
┌────────────────────────────────────────────────────────────────┐
│                    Data Layer                                   │
│                                                                  │
│   PostgreSQL        MinIO            Qdrant         Redis       │
│   (users, projects, (uploads,        (dense +       (Celery     │
│    jobs, QA pairs,   extracted text,  sparse         broker)    │
│    LangGraph         images)          vectors)                  │
│    checkpoints)                                                 │
└────────────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────────┐
│           External APIs (configurable per project)             │
│   LLM:        OpenAI · Anthropic · Google · Groq · Mistral     │
│               Cohere · Together · OpenRouter · Azure · Ollama  │
│   Embeddings: OpenAI · Google · Cohere · VoyageAI · Jina       │
│               Ollama                                           │
│   Reranker:   Cohere · Jina · Ollama                           │
└────────────────────────────────────────────────────────────────┘
```

### 2.2 Two-Phase Design

| Phase | Purpose | Orchestration |
|---|---|---|
| Phase 1: Document Processing | Ingest, extract, chunk, embed, store | Celery (heavy, blocking tasks) |
| Phase 2: QA Generation | Generate, retrieve, validate, review, export | LangGraph (agentic, async LLM calls) |

Redis is only the Celery broker. PostgreSQL holds all application state and the LangGraph checkpoints (in a separate `langgraph` schema). MinIO holds files: uploads, extracted text/markdown, and extracted images.

### 2.3 Provider & Credential System

Every LLM call in the platform — contextualization, embedding, question generation, answer generation, validation, reranking — goes through **LiteLLM**, one library with the same call for 100+ providers. The model id (`anthropic/claude-sonnet-4-5`, `ollama_chat/llama3.1`, ...) picks the provider. There are no hardcoded API keys anywhere in the code. Everything provider-specific sits in one registry (`web_api/services/providers.py`); adding a provider is one entry.

**Credential store:** Credentials are global, named connections (e.g. "OpenAI – team key") that every admin can see and manage. Secrets (API keys) are encrypted at rest with Fernet (AES-128); plain settings (Ollama URL, Azure endpoint and API version) are stored as-is and shown on the connection's configuration page. The only secret in the environment is the `ENCRYPTION_KEY`. Keys are never returned by any API response. A credential that is still attached to a project cannot be deleted. Ollama and Azure connections also hold **registered models** (downloaded Ollama models, Azure deployment names), because no catalog can know those.

**Model discovery:** The model dropdown comes from LiteLLM's model catalog (type, context size, vision support, embedding size), checked against the provider's live model list where the provider has one. Ollama connections sync their list from the server itself. An "Other…" option allows any model name.

**Per-project model config:** Project owners attach a credential + model to each pipeline stage. The seven stages are: `question_generator`, `answer_generator`, `validator`, `meta_agent`, `embedder`, `reranker`, and `vision` (image captioning). Each stage needs one capability (chat, embedding, rerank or vision); a catalog model of the wrong kind is rejected. The embedder's vector size is measured when it is saved, and the embedder is locked once the first document has been embedded.

**Validation:** Before a stage is saved, the platform makes one tiny real call with its model to catch bad model names or keys immediately, not halfway through a job.

**Worker access to keys:** Celery tasks carry only IDs. The worker fetches the project's models and keys from an internal endpoint once per job and keeps them in memory, so keys never sit in Redis.

**Workers:** When a Celery task is dispatched, the web API fetches and decrypts the relevant credentials and passes them in the task payload. Workers use the passed credentials, falling back to environment variables if none are provided. Workers never talk to PostgreSQL or MinIO directly: they fetch files and report results through web API `/internal/*` and `/webhooks/*` routes, authenticated with a shared `X-Internal-Token`.

---

## 3. Phase 1 — Document Processing Pipeline

### 3.1 Document Categories

The platform currently supports two document categories with specialized processing strategies:

| Category | Input | Processing Strategy |
|---|---|---|
| `FICTION` | Novels, books, prose PDFs | PyMuPDF text extraction |
| `ACADEMIC` | Research papers, technical PDFs | Marker CLI → Markdown + vision captioning |

### 3.2 Fiction Pipeline

```
PDF Upload
    ↓
PyMuPDF → raw text extraction
    ↓
Stored → MinIO (text) + PostgreSQL `extractions` row (key + stats)
    ↓
Chonkie hierarchical chunking
    ├─ Parent chunks: 30,000 tokens, 5,000 overlap
    └─ Child chunks: 800 tokens, no overlap
    ↓
Contextualization with the project's meta_agent model (any provider, via LiteLLM)
    └─ Each child chunk gets 50–200 token context description
       generated from its parent window
    ↓
Embedder from project config (dimension = embedding_dim)
    ↓
BM25 sparse vector generation
    ↓
Qdrant → the project's collection (project_<id>)
    └─ Each point: dense vector + sparse vector + payload
```

### 3.3 Academic Pipeline

```
PDF Upload
    ↓
Marker CLI (PDF → Markdown)
    ├─ use_llm: true (Google Gemini backend)
    ├─ force_ocr: true
    ├─ redo_inline_math: true
    └─ Extracts: markdown text + image files + metadata.json
    ↓
Project's vision model → image captioning
    └─ Each image: surrounding markdown context (300 chars)
       + image file → AI-generated description
    ↓
Image replacement in markdown
    ├─ Before: ![alt](image.png)
    └─ After: **[IMAGE: alt]**\n{description}\n
    ↓
Stored → MinIO + PostgreSQL
    └─ text.md (original markdown)       → MinIO
    └─ enriched.md (image descriptions)  → MinIO
    └─ images                            → MinIO
    └─ keys, stats, image descriptions   → `extractions` / `extracted_images` rows
    ↓
Smart markdown-aware chunking
    ├─ Respects heading hierarchy, code blocks, math blocks, tables
    ├─ Parent chunks: 30,000 tokens, 5,000 overlap
    └─ Child chunks: 800 tokens, no overlap
    ↓
Contextualization → embeddings (project's models) → BM25
    ↓
Qdrant → the project's collection (project_<id>)
```

### 3.4 Chunking Strategy

The chunking design is hierarchical with a clear separation of roles:

| Level | Size | Overlap | Purpose |
|---|---|---|---|
| Parent chunk | 30,000 tokens | 5,000 tokens | Context window for LLM during contextualization and QA generation |
| Child chunk | 800 tokens | 0 | Retrieval unit — what gets embedded and searched in Qdrant |

Each child chunk goes through contextualization:
- The parent chunk (30k window) is given to GPT-4o-mini
- LLM generates a 50–200 token description of the broader context
- `combined_text = context_description + original_child_text`
- `combined_text` is what gets embedded — not the raw child text

This produces embeddings that understand the chunk *in context*, significantly improving retrieval quality.

### 3.5 Qdrant Storage Schema

Each point stored in Qdrant:

```
Point:
  id: UUID
  vectors:
    dense:  List[float]  # embedding_dim from the project config
    sparse: SparseVector # BM25 indices + values
  payload:
    original_text:       str
    context_description: str
    combined_text:       str
    document_id:         str
    project_id:          str   ← used for filtering during QA retrieval
    chunk_index:         int
    parent_context_id:   str
    start_index:         int
    end_index:           int
    token_count:         int
    metadata:            dict
```

### 3.6 Infrastructure for Phase 1

- **Celery workers**: separate process pool, Redis broker
- **Webhook callbacks**: workers POST to `/webhooks/processing-complete` when done
- **Temp file management**: `/tmp/celery_files/{task_id}/` — auto-cleaned after completion
- **File streaming**: files < 5MB via base64, files > 5MB via chunked stream
- **Monitoring**: Flower dashboard on port 5555

---

## 4. Phase 2 — QA Generation Pipeline

### 4.1 Design Decision: LangGraph over Celery

QA generation is fundamentally different from document processing:

| Document Processing | QA Generation |
|---|---|
| Few heavy blocking tasks (Marker takes minutes) | Many small async LLM calls (seconds each) |
| CPU/IO bound | Network bound (API calls) |
| Needs worker isolation | Can run as async background tasks in web_api |
| Simple linear pipeline | Stateful graph with branching and human interrupts |
| No human-in-the-loop | Human review is a core part of the flow |

LangGraph is used for QA generation because:
- Built-in `interrupt()` for human-in-the-loop (the human review queue)
- Built-in checkpointing to PostgreSQL (`AsyncPostgresSaver`, same database) — enables pause/resume
- Conditional edges map directly to the score-based routing logic
- `Send` API enables fan-out to process hundreds of QA chunks in parallel
- State machine is the natural model for the QA pair lifecycle

Celery remains exclusively for document processing (Phase 1).

### 4.2 QA-Specific Chunking

Before question generation begins, documents are re-chunked with different parameters than the retrieval chunks:

| | Retrieval Chunks (Phase 1) | QA Generation Chunks (Phase 2) |
|---|---|---|
| Size | 800 tokens (child) | 60,000 tokens |
| Overlap | 5,000 tokens (parent) | None |
| Purpose | Semantic search target | Question generation input |
| Storage | Qdrant (permanent) | PostgreSQL `qa_chunks` (temporary) |
| Source | Original PDF | Extracted text already in MinIO |

**Why 60,000 tokens?** Standard LLM context windows are ~128k tokens. Half is reserved for the prompt template, generated questions, and system instructions. 60k tokens is dense enough to generate multiple diverse questions while staying within context limits.

**Source for academic documents**: `enriched_markdown` (with image descriptions inlined) — not `markdown_text`. This gives the question generator awareness of visual content.

No re-processing of PDFs. Content is read directly from the extracted text in MinIO (located via the `extractions` table).

### 4.3 LangGraph Graph Structure

```
                    ┌─────────────┐
                    │  chunk_node │
                    │             │
                    │ Fetch from  │
                    │ MinIO,      │
                    │ rechunk to  │
                    │ 60k tokens  │
                    └──────┬──────┘
                           │ Send API (fan-out per QA chunk)
              ┌────────────┼────────────┐
              ▼            ▼            ▼
    ┌──────────────────────────────────────┐
    │         question_gen_node            │
    │                                      │
    │  question_generator_prompt           │
    │  + 60k chunk content                 │
    │  + config (N questions, types)       │
    │                                      │
    │  Output: N questions                 │
    │  Types: FACTUAL | INFERENTIAL        │
    │          ANALYTICAL | COT            │
    └──────────────┬───────────────────────┘
                   │ per question
                   ▼
    ┌──────────────────────────────────────┐
    │          answer_gen_node             │
    │                                      │
    │  1. Hybrid search Qdrant             │
    │     (filter: project_id)             │
    │  2. Rerank top-K results             │
    │  3. answer_generator_prompt          │
    │     + retrieved chunks               │
    │     + user_reflection_comment        │
    │       (if retry)                     │
    │  4. Generate answer                  │
    │     (+ <think> block if COT)         │
    └──────────────┬───────────────────────┘
                   │
                   ▼
    ┌──────────────────────────────────────┐
    │          validate_node               │
    │                                      │
    │  validator_prompt + QA pair          │
    │  Output: score (1.0–10.0)            │
    │          + reasoning string          │
    └──────────────┬───────────────────────┘
                   │
                   ▼
    ┌──────────────────────────────────────┐
    │         route_edge (conditional)     │
    └──────┬──────────────┬───────────────┬┘
           │              │               │
     score ≥ 9       score 5–8       score < 5
           │              │               │
           ▼              ▼               ▼
    AUTO_ACCEPTED   interrupt()     AUTO_REJECTED
    → dataset       human review
                         │
              ┌──────────┴──────────┐
              │                     │
           approve              reject
              │                     │
       HUMAN_ACCEPTED        HUMAN_REJECTED
       → dataset
              │
           comment
              │
       PENDING_REGENERATION
              │
        retry_count < 3?
           ↓ yes          ↓ no
    back to          AUTO_REJECTED
    answer_gen_node
    (with comment)
```

### 4.4 The Three Agents

Each agent has a dedicated system prompt configured per project before any job runs. Each agent also uses the provider and model configured for its stage in the project's model config (see §2.3) — the same question generation prompt can run against GPT-4o, Claude 3.5 Sonnet, or a local Ollama model without any code change.

#### Question Generator
- **Input**: 60k QA chunk + `question_generator_prompt` + job config
- **Config**: questions per chunk (N), question types to generate
- **Output**: list of N questions, each tagged with type
- **Question types**:
  - `FACTUAL` — direct knowledge recall ("What is X?")
  - `INFERENTIAL` — requires connecting information ("Why did X lead to Y?")
  - `ANALYTICAL` — deeper reasoning ("What are the implications of X?")
  - `COT` — chain-of-thought format requiring step-by-step reasoning

#### Answer Generator
- **Input**: question + reranked retrieved chunks + `answer_generator_prompt`
- **If retry**: `user_reflection_comment` appended as additional instruction
- **Output**: answer string; for COT type, also a `reasoning` string
- **CoT answer format**: `<think>\n{step-by-step reasoning}\n</think>\n{final answer}`

#### Validator
- **Input**: question + answer + `validator_prompt`
- **Output**: `score` (float 1.0–10.0) + `reasoning` (string explaining the score)
- The reasoning is stored and shown to human reviewers when a pair enters the review queue

### 4.5 Skill Configuration System

Before a QA job can start, the project owner must configure and approve a Skill Config. This is a set of system prompts that defines how each agent behaves for this specific project.

**The meta-agent** — a separate LLM interaction (not a Celery task, not a LangGraph node) that lives in `web_api` as a service. The project owner describes what they want in plain language. The meta-agent drafts system prompts for all three agents. The owner reviews, edits, and iterates until satisfied, then explicitly approves.

Approval locks the skill config and unlocks job creation.

```
Owner describes dataset goal in plain language
    ↓
Meta-agent drafts:
    ├─ question_generator_prompt
    ├─ answer_generator_prompt
    └─ validator_prompt
    + base_context (auto-built from project type + document list)
    ↓
Owner reviews → edits → re-generates if needed
    ↓
Owner approves → SkillConfig.approved = true
    ↓
Job can now be created and started
```

Skill configs are versioned per project. A project can have multiple approved skill config versions. Each QA job references a specific skill config version. This means you can run two jobs on the same project with different prompting strategies and produce two different datasets.

### 4.6 Answer Retrieval Detail

```
Question string
    ↓
Hybrid search in Qdrant
    ├─ Dense vector search (semantic similarity)
    ├─ Sparse BM25 search (keyword match)
    └─ Payload filter: { project_id: <current_project_id> }
       → searches across ALL documents in the project
       → never crosses project boundaries
    ↓
Top-K results returned (e.g. K=10)
    ↓
Reranker (FlashRank or Cohere)
    └─ Cross-encoder reranks by relevance to the question
    ↓
Top results (e.g. top 3–5) passed to answer_gen_node
    ↓
Stored on QAPairModel:
    ├─ retrieved_chunk_ids: List[str]   (Qdrant point IDs)
    └─ retrieval_scores: List[float]    (post-rerank scores)
```

### 4.7 Checkpointing and Pause/Resume

LangGraph persists graph state to PostgreSQL on every node transition using an `AsyncPostgresSaver` checkpointer (tables in the `langgraph` schema, created by `web_api.scripts.setup_checkpointer`; available at runtime as `app.state.checkpointer`). Each job is identified by a unique `thread_id` (the `job_id`).

- **Pause**: `POST /qa-jobs/{id}/pause` → sets `qa_jobs.status = PAUSED`. LangGraph state is already persisted.
- **Resume**: `POST /qa-jobs/{id}/resume` → calls `graph.ainvoke` with the same `thread_id`. LangGraph reads the checkpoint and continues.
- **Server restart**: no data loss. PostgreSQL holds the graph state. Job restarts from last checkpoint.
- **QA pair progress**: pairs already in terminal states (`AUTO_ACCEPTED`, `HUMAN_ACCEPTED`, `AUTO_REJECTED`, `HUMAN_REJECTED`) are skipped on resume.

### 4.8 Concurrency Control

Processing a large project generates thousands of LLM calls. A semaphore in `answer_gen_node` caps concurrent OpenAI API calls (same pattern as existing `contextualizer.py` in Phase 1).

```python
semaphore = asyncio.Semaphore(10)  # max 10 concurrent answer gen calls

async def answer_gen_node(state):
    async with semaphore:
        # LLM call here
```

---

## 5. Human-Guided Workflow

### 5.1 The Human Review Queue

When a QA pair scores 5–8, the LangGraph graph pauses via `interrupt()` and the pair enters `PENDING_HUMAN_REVIEW` status. The web API exposes endpoints for workers and project owners to act on these pairs.

```
GET /qa-pairs/review/{project_id}
    → paginated list of PENDING_HUMAN_REVIEW pairs
    → includes: question, answer, validation_score, validation_reasoning,
                retrieved_chunks used, source document, retry_count

POST /qa-pairs/{id}/approve     → HUMAN_ACCEPTED → dataset
POST /qa-pairs/{id}/reject      → HUMAN_REJECTED
POST /qa-pairs/{id}/comment     → body: { comment: "..." }
                                → status: PENDING_REGENERATION
                                → graph resumes → answer_gen_node
                                → user_reflection_comment injected into prompt
```

### 5.2 Retry Mechanism

```python
# On QAPairModel
user_reflection_comment: Optional[str]   # present = retry mode
retry_count: int                         # increments each retry

# In answer_gen_node
if state.qa_pair.user_reflection_comment:
    prompt += f"\n\nHuman feedback: {state.qa_pair.user_reflection_comment}"
    prompt += "\nPlease revise the answer addressing this feedback."

# After validation
if score < threshold and retry_count >= 3:
    status = AUTO_REJECTED  # force reject after max retries
```

### 5.3 DPO Pair Generation

Since multiple answer attempts are generated per question (original + retries), the system naturally produces preference pairs:
- `chosen`: the final accepted answer (highest quality)
- `rejected`: an earlier lower-scored attempt

The `QAPairModel` tracks `retry_count` and links attempts, enabling DPO export automatically from the audit trail.

---

## 6. Role & Permission System

### 6.1 Role Hierarchy

```
Application Level:
│
├── Admin
│     Full system access. Manage all users, all projects,
│     view system stats, manage infrastructure config.
│     Set and rotate API credentials for all LLM/embedding/reranking providers.
│
└── User  (base role for all registered accounts)
      Can create projects, upload documents, manage their own content.
      When a User creates a project → they become that project's Owner.
      When a User is invited to a project → they become a Worker.

Project Level (scoped to one project):
│
├── Project Owner
│     Created the project. Full project control:
│     - Manage documents and processing
│     - Configure and approve skill configs
│     - Create and manage QA jobs (start, pause, resume)
│     - Add and remove workers
│     - Review QA pairs (same as worker)
│     - Export dataset
│
└── Worker
      Invited by project owner. Single responsibility:
      - Review PENDING_HUMAN_REVIEW QA pairs
      - Approve / reject / comment on pairs
      No access to project settings, jobs, or documents.
```

A user can simultaneously be a Project Owner of one project and a Worker in another.

### 6.2 Auth

- Admins invite users by email; the user sets username + password via a one-time setup link (token stored as a SHA-256 hash, 24 h expiry)
- JWT access tokens (`python-jose`), Argon2 password hashing (`argon2-cffi`)
- Every request reloads the user, so deactivation takes effect immediately
- Project-level role checked per endpoint via project membership lookup; admins can access every project; non-members get 404

---

## 7. Data Models

PostgreSQL tables via SQLAlchemy 2 (async) with Alembic migrations — see `web_api/db/models/` for the source of truth. All ids are UUIDs; all timestamps are `TIMESTAMPTZ`; enums are stored as text with a CHECK constraint. Large content (files, extracted text) lives in MinIO; tables store the object keys.

### 7.1 Implemented Tables

#### `users`
```
email:                   str        unique, stored lowercase
username:                str?       unique, set during account setup
password_hash:           str?       Argon2, set during account setup
app_role:                AppRole    admin | user
is_active:               bool
must_change_password:    bool       true until the invite is accepted
setup_token_hash:        str?       sha256 of the one-time setup token
setup_token_expires_at:  datetime?
```

#### `email_events`
```
user_id → users (cascade), resend_id (unique), kind (invite), sent_at
```

#### `projects`
```
title, description
data_type:          Datatype   fiction | academic
created_by:         → users (restrict)
embedding_locked:   bool       true once the first document is embedded
```

#### `project_members`
```
PK (project_id → projects cascade, user_id → users cascade)
role:      ProjectRole   owner | worker
added_by:  → users (set null)
```
The creator is added as `owner` in the same transaction; a project always keeps at least one owner.

#### `documents`
```
project_id:     → projects (cascade)
original_name, storage_key (MinIO, unique), file_type (pdf | images),
content_type, size_bytes
status:         uploaded | queued | processing | completed | failed
error, uploaded_by, processed_at
```

#### `provider_credentials`
```
name:              str        unique, e.g. "OpenAI - team key"
provider:          ModelProvider
encrypted_fields:  jsonb      field_name → Fernet token (api_key, endpoint, ...)
created_by:        → users
```
> Global and admin-managed. Keys are write-only from the API.

#### `project_stage_models`
```
PK (project_id → projects cascade, stage)
credential_id:  → provider_credentials (restrict: in-use credentials can't be deleted)
model_name, embedding_dim? (embedder only, measured by a test call)
```

#### `credential_models`
```
credential_id → provider_credentials (cascade), name (Ollama tag / Azure deployment), base_model?,
capabilities[], embedding_dim?, context_window?, status (untested | ok | failed), last_error?, last_checked_at?
UNIQUE (credential_id, name)
```
> Only for Ollama / Azure connections, whose models can't be looked up in a catalog.

#### `processing_jobs` / `job_documents`
```
processing_jobs:  project_id, celery_task_id, status (queued | running | completed | partial | failed),
                  error, requested_by, started_at, finished_at
job_documents:    PK (job_id, document_id), status, error
```

#### `extractions` / `extracted_images`
```
extractions:       document_id (PK), text_key, enriched_key?, char_count, image_count, metadata (jsonb)
extracted_images:  document_id, filename, storage_key, description, position
```

#### `chunks`
```
document_id → documents (cascade), chunk_index, qdrant_point_id (uuid)
UNIQUE (document_id, chunk_index)   — a retried document replaces its rows
```

### 7.2 Planned Tables (Phase 2)

Types below are logical; ids become `UUID` foreign keys when implemented.

#### `SkillConfigModel` — table: `skill_configs`
```
project_id:                  UUID
version:                     int         increments on each save
question_generator_prompt:   str
answer_generator_prompt:     str
validator_prompt:            str
base_context:                str         auto-built: project type + doc summaries
questions_per_chunk:         int
question_types:              List[QuestionType]   FACTUAL | INFERENTIAL | ANALYTICAL | COT
approved:                    bool
approved_by:                 Optional[UUID]
approved_at:                 Optional[datetime]
created_at:                  datetime
```

#### `QAJobModel` — table: `qa_jobs`
```
project_id:            UUID
skill_config_id:       UUID
skill_config_version:  int
status:                QAJobStatus
                         CREATED | CHUNKING | RUNNING |
                         PAUSED | COMPLETED | FAILED
created_by:            UUID
total_chunks:          int
processed_chunks:      int         checkpoint: QA chunks completed
total_questions:       int
auto_accepted:         int
pending_human:         int
auto_rejected:         int
human_accepted:        int
human_rejected:        int
created_at:            datetime
updated_at:            datetime
```

#### `QAChunkModel` — table: `qa_chunks`
```
job_id:         UUID
project_id:     UUID
document_id:    UUID
chunk_index:    int
content:        str           60,000 token chunk from extracted content
token_count:    int
status:         QAChunkStatus    PENDING | PROCESSING | COMPLETED | FAILED
source_type:    Datatype         FICTION | ACADEMIC
```
> Ephemeral — safe to delete after job completes.

#### `QAPairModel` — table: `qa_pairs`
```
job_id:                   UUID
project_id:               UUID
document_id:              UUID
qa_chunk_id:              UUID

question:                 str
question_type:            QuestionType    FACTUAL | INFERENTIAL | ANALYTICAL | COT

answer:                   str
reasoning:                Optional[str]   populated for COT type

retrieved_chunk_ids:      List[str]       Qdrant point IDs used
retrieval_scores:         List[float]     post-rerank relevance scores

validation_score:         float           1.0 – 10.0
validation_reasoning:     str             validator LLM explanation

status:                   QAPairStatus
                            PENDING_ANSWER
                            PENDING_VALIDATION
                            AUTO_ACCEPTED          score ≥ 9
                            PENDING_HUMAN_REVIEW   score 5–8
                            HUMAN_ACCEPTED
                            AUTO_REJECTED          score < 5
                            HUMAN_REJECTED
                            PENDING_REGENERATION   sent back with comment

user_reflection_comment:  Optional[str]   present = retry mode
retry_count:              int             max 3, then force AUTO_REJECTED
reviewed_by:              Optional[UUID]
reviewed_at:              Optional[datetime]
created_at:               datetime
updated_at:               datetime
```

### 7.3 Enums

```python
class AppRole(str, Enum):
    ADMIN = "admin"
    USER = "user"

class ProjectRole(str, Enum):
    OWNER = "owner"
    WORKER = "worker"

class Datatype(str, Enum):
    FICTION = "fiction"
    ACADEMIC = "academic"

class QuestionType(str, Enum):
    FACTUAL = "FACTUAL"
    INFERENTIAL = "INFERENTIAL"
    ANALYTICAL = "ANALYTICAL"
    COT = "COT"

class QAJobStatus(str, Enum):
    CREATED = "CREATED"
    CHUNKING = "CHUNKING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class QAChunkStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class QAPairStatus(str, Enum):
    PENDING_ANSWER = "PENDING_ANSWER"
    PENDING_VALIDATION = "PENDING_VALIDATION"
    AUTO_ACCEPTED = "AUTO_ACCEPTED"
    PENDING_HUMAN_REVIEW = "PENDING_HUMAN_REVIEW"
    HUMAN_ACCEPTED = "HUMAN_ACCEPTED"
    AUTO_REJECTED = "AUTO_REJECTED"
    HUMAN_REJECTED = "HUMAN_REJECTED"
    PENDING_REGENERATION = "PENDING_REGENERATION"

class DatasetFormat(str, Enum):
    CHAT = "CHAT"
    DPO = "DPO"
    COT = "COT"
```

---

## 8. API Design

Implemented endpoints are marked ✅; the rest are planned. The live, exact contract is always at `/docs` (Swagger).

### 8.1 Users & Auth ✅
```
POST   /users                               Invite a user by email (admin)
GET    /users                               List users (admin)
POST   /users/resend-invite                 New setup link; old one stops working (admin)
PATCH  /users/{user_id}/active              Activate / deactivate (admin)
POST   /users/setup                         Accept invite: token + username + password
POST   /users/login                         OAuth2 form (username = email) → JWT
GET    /users/me                            Current user
```

### 8.2 Projects & Members ✅
```
POST   /projects                            Create (caller becomes owner)
GET    /projects                            Projects I'm a member of (admin: all)
GET    /projects/{id}
PATCH  /projects/{id}                       Owner
DELETE /projects/{id}                       Owner — cascades documents, config, MinIO files

GET    /projects/{id}/members
POST   /projects/{id}/members               Add by email (owner)
PATCH  /projects/{id}/members/{user_id}     Change role (owner)
DELETE /projects/{id}/members/{user_id}     Remove (owner; last owner can't be removed)
```

### 8.3 Documents ✅
```
POST   /projects/{id}/documents             Upload one or more files (all-or-nothing, content-checked)
GET    /projects/{id}/documents
GET    /documents/{document_id}
DELETE /documents/{document_id}
```

### 8.4 Document Processing ✅
```
POST   /projects/{id}/processing-jobs       Queue documents (no ids = all uploaded/failed)
GET    /projects/{id}/processing-jobs       List jobs
GET    /processing-jobs/{job_id}            Job + per-document status
GET    /projects/{id}/extractions           Extraction summaries
GET    /documents/{document_id}/extraction  Extracted text / enriched markdown
```

### 8.5 Skill Configuration
```
POST   /skills/draft                       Meta-agent drafts prompts from description
                                           Body: { project_id, description }
                                           Returns: SkillConfigModel (draft)

PUT    /skills/{skill_id}                  Edit draft
GET    /skills/{skill_id}                  Get skill config
GET    /skills/project/{project_id}        List all versions for project

POST   /skills/{skill_id}/approve          Approve and lock skill config (owner only)
                                           Unlocks QA job creation
```

### 8.6 QA Jobs
```
POST   /qa-jobs/                           Create QA job
                                           Body: { project_id, skill_config_id,
                                                   questions_per_chunk, question_types }

POST   /qa-jobs/{id}/start                 Start processing (triggers LangGraph graph)
POST   /qa-jobs/{id}/pause                 Checkpoint and pause
POST   /qa-jobs/{id}/resume                Resume from checkpoint

GET    /qa-jobs/{id}                       Job status + progress counters
GET    /qa-jobs/project/{project_id}       List all jobs for project
```

### 8.7 Human Review
```
GET    /qa-pairs/review/{project_id}       Paginated PENDING_HUMAN_REVIEW pairs
                                           Query params: ?page=1&limit=20&document_id=...
                                           Returns: question, answer, score, reasoning,
                                                    retrieved chunks, source doc, retry_count

POST   /qa-pairs/{id}/approve              Mark HUMAN_ACCEPTED
POST   /qa-pairs/{id}/reject               Mark HUMAN_REJECTED
POST   /qa-pairs/{id}/comment              Body: { comment: "..." }
                                           → PENDING_REGENERATION
                                           → graph resumes with comment injected
```

### 8.8 Dataset
```
GET    /dataset/{project_id}               Dataset summary
                                           Returns: { total_pairs, by_type,
                                                      auto_accepted, human_accepted,
                                                      available_formats }

POST   /dataset/{project_id}/export        Body: { format: CHAT | DPO | COT }
                                           Returns: JSONL file download
```

### 8.9 Credentials ✅
```
GET    /credentials/schema              Required fields per provider (drives UI forms)
GET    /credentials/options             id + name + provider, for any user picking a credential
GET    /credentials                     All credentials + projects using them (admin). No key values.
POST   /credentials                     Create: { name, provider, fields } (admin)
GET    /credentials/{id}                (admin)
PATCH  /credentials/{id}                Rename / rotate keys (admin)
DELETE /credentials/{id}                409 if still attached to a project (admin)
```

### 8.10 Model Configuration ✅
```
GET    /projects/{id}/model-config               Current stage assignments
PUT    /projects/{id}/model-config               Upsert stages (owner)
                                                 Body: { stages: { embedder: { credential_id, model_name }, ... } }
                                                 Each stage is test-called; embedding size is measured
DELETE /projects/{id}/model-config/{stage}       (owner)
POST   /projects/{id}/model-config/validate      Live ping per stage; saves nothing
```

### 8.11 Internal (worker-facing, `X-Internal-Token` required) ✅
```
GET    /internal/files/{id}/metadata             Also marks the document as processing
GET    /internal/files/{id}/base64               Files < 5 MB
GET    /internal/files/{id}/stream               Any size
POST   /internal/extracted/fiction
POST   /internal/extracted/academic/images/{document_id}
POST   /internal/extracted/academic
```

### 8.12 Webhooks (worker-facing, `X-Internal-Token` required) ✅
```
POST   /webhooks/processing-complete       Chunk → Qdrant point links; document + job status
POST   /webhooks/processing-failed         Marks the document failed with the stage + error
```

---

## 9. Infrastructure & Tech Stack

### 9.1 Full Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Web framework | FastAPI | HTTP API, background tasks |
| Database | PostgreSQL 17 (Docker) | All application state |
| ORM / driver | SQLAlchemy 2 (async) + psycopg 3 | Typed models, async access |
| Migrations | Alembic | Versioned schema changes |
| Object storage | MinIO | Uploads, extracted text, extracted images |
| Task queue | Celery | Document processing (Phase 1) |
| Message broker | Redis 7 | Celery broker |
| QA orchestration | LangGraph | Agentic QA pipeline (Phase 2) |
| Graph checkpointing | PostgreSQL (`langgraph-checkpoint-postgres`) | Pause/resume state |
| Vector database | Qdrant | Dense + sparse vector search |
| PDF text extraction | PyMuPDF | Fiction documents |
| PDF → Markdown | Marker CLI | Academic documents |
| Chunking | Chonkie | Both pipelines + QA chunks |
| LLM | OpenAI · Anthropic · Google · Groq · Mistral · Cohere · Together · OpenRouter · Azure OpenAI · Ollama | All LLM stages, configurable per project |
| Embeddings | OpenAI · Google · Cohere · VoyageAI · Jina · Ollama | Dense vector generation, configurable per project |
| Vision / Marker LLM | Google Gemini | Image captioning, Marker backend |
| Reranker | Cohere · Jina · Ollama | Retrieved chunk reranking, configurable per project |
| Credential encryption | cryptography (Fernet / AES-128) | Encrypt provider API keys at rest |
| Authentication | python-jose + argon2-cffi | JWT tokens, password hashing |
| Email | Resend | Invite / setup-password emails |
| Validation / config | Pydantic v2 + pydantic-settings | Schemas, typed settings from `.env` |
| Monitoring | Flower (port 5555) | Celery task dashboard |
| Language | Python 3.11+ | |

### 9.2 Docker Compose Services

```yaml
services:
  postgres:   # 127.0.0.1:5433 — app DB + LangGraph checkpoints (roles: synth_owner, synth_app)
  redis:      # port 6379 — Celery broker
  qdrant:     # port 6333 — vector database
  minio:      # ports 9000 (API) / 9001 (console) — object storage
  flower:     # port 5555 — Celery monitoring
```

Two database roles: `synth_owner` owns the schema and runs migrations; `synth_app` (used by the running API) can only read and write data.

### 9.3 Directory Structure

```
├── web_api/                 FastAPI service (own uv project)
│   ├── main.py              app, lifespan, router registration
│   ├── core/                settings (pydantic-settings), Celery client
│   ├── db/                  SQLAlchemy base, session, models/, LangGraph checkpointer
│   ├── alembic/             migrations
│   ├── data_models/         Pydantic request/response schemas + enums
│   ├── deps/                auth, project access, internal-token guard
│   ├── routers/             HTTP layer
│   ├── services/            business logic
│   ├── scripts/             create_admin, setup_checkpointer
│   ├── tests/               pytest (test database, faked email/MinIO/Celery)
│   └── qa_pipeline/         LangGraph graph, nodes, state               [PLANNED]
│
├── workers/                 Celery document processing (own uv project)
│   ├── tasks/               orchestrator, fiction_processor, academic_processor
│   ├── services/            fetch, extract, chunk, contextualize, embed, BM25, Qdrant
│   └── utils/               temp files, webhook notifier
│
├── front-end/               React + Vite UI
├── infra/postgres/init/     first-start role/database setup
├── docs/                    design (this file), development guide, brand kit
└── docker-compose.yml
```

---

## 10. Export Formats

### 10.1 Chat Format (JSONL)
Standard instruction fine-tuning. Compatible with OpenAI, Mistral, LLaMA fine-tuning pipelines.

```json
{"messages": [
  {"role": "system", "content": "You are a knowledgeable assistant..."},
  {"role": "user", "content": "What is the significance of X in the context of Y?"},
  {"role": "assistant", "content": "X is significant because..."}
]}
```

### 10.2 CoT Format (JSONL)
Chain-of-thought reasoning format. Teaches the model to reason before answering. Uses `<think>` block convention compatible with modern reasoning models.

```json
{"messages": [
  {"role": "system", "content": "Think step by step before providing your answer."},
  {"role": "user", "content": "Why did X lead to Y?"},
  {"role": "assistant", "content": "<think>\nStep 1: ...\nStep 2: ...\nTherefore...\n</think>\nThe reason X led to Y is..."}
]}
```

### 10.3 DPO Format (JSONL)
Preference pairs for Direct Preference Optimization. Teaches the model to prefer higher-quality responses. Chosen/rejected pairs come from the retry audit trail — original lower-scored answer vs final accepted answer.

```json
{
  "prompt": [
    {"role": "user", "content": "What are the main findings of this study?"}
  ],
  "chosen": [
    {"role": "assistant", "content": "The study found three main results: ..."}
  ],
  "rejected": [
    {"role": "assistant", "content": "The study had some findings about..."}
  ]
}
```

---

## 11. Open Decisions

The following design choices are not yet finalized:

| Decision | Options | Recommendation |
|---|---|---|
| **Reranker** | Cohere, Jina, or Ollama — selected per project via model config | Default to Cohere if available; Ollama for local/offline setups |
| **DPO pair source** | (A) Deliberately generate 2 answers per question, or (B) use retry audit trail (original = rejected, final = chosen) | Option B — free, no extra LLM calls, natural quality gap |
| **Retry cap** | Max retries before force-rejecting a QA pair | 3 retries |
| **Academic QA source** | `markdown_text` vs `enriched_markdown` for QA chunks | `enriched_markdown` — image descriptions provide richer context for question generation |
| **Validator model** | Same `gpt-4o-mini` as other agents, or a separate/stronger model | `gpt-4o-mini` for consistency and cost; can be overridden per skill config |

---

*Document version: 1.1 — updated 2026-06-07: added provider & credential system (§2.3, §8.9–8.10)*
