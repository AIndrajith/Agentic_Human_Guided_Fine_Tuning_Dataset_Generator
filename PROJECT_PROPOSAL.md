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
│   MongoDB          Qdrant              Redis                    │
│   (documents,      (dense +            (Celery broker +         │
│    QA pairs,       sparse vectors)     LangGraph checkpoints)   │
│    datasets)                                                    │
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

The same Redis instance serves as both the Celery broker and the LangGraph checkpoint store.

### 2.3 Provider & Credential System

Every LLM call in the platform — contextualization, embedding, question generation, answer generation, validation, reranking — routes through a single factory (`get_llm_client`) that resolves the right provider and model at runtime. There are no hardcoded API keys anywhere in the code.

**Credential store:** An admin sets API keys through the platform's Credentials configuration page. Keys are encrypted at rest using Fernet symmetric encryption (AES-128) before being stored in MongoDB. The only secret that lives in the environment is the `ENCRYPTION_KEY` used for encryption/decryption. Keys are never returned by any API response — only a status of which fields are present is exposed.

**Per-project model config:** Project owners configure which provider and model to use for each pipeline stage. The six configurable stages are: `question_generator`, `answer_generator`, `validator`, `meta_agent`, `embedder`, and `reranker`. The capability map enforces valid combinations — for example, Anthropic cannot serve the embedder stage since it has no embedding API.

**Validation:** Before saving a config, the platform pings each stage's provider with a minimal request to catch bad model names or missing credentials immediately, not halfway through a job.

**Workers:** When a Celery task is dispatched, the web API fetches and decrypts the relevant credentials and passes them in the task payload. Workers use the passed credentials, falling back to environment variables if none are provided (backwards-compatible).

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
Stored → MongoDB (ExtractedFictionModel)
    ↓
Chonkie hierarchical chunking
    ├─ Parent chunks: 30,000 tokens, 5,000 overlap
    └─ Child chunks: 800 tokens, no overlap
    ↓
GPT-4o-mini contextualization
    └─ Each child chunk gets 50–200 token context description
       generated from its parent window
    ↓
OpenAI text-embedding-3-large (1536-dim)
    ↓
BM25 sparse vector generation
    ↓
Qdrant → collection: fiction_chunks
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
Gemini Vision → image captioning
    └─ Each image: surrounding markdown context (300 chars)
       + image file → AI-generated description
    ↓
Image replacement in markdown
    ├─ Before: ![alt](image.png)
    └─ After: **[IMAGE: alt]**\n{description}\n
    ↓
Stored → MongoDB (ExtractedAcademicModel)
    └─ markdown_text (original)
    └─ enriched_markdown (with image descriptions)
    └─ images metadata list
    ↓
Smart markdown-aware chunking
    ├─ Respects heading hierarchy, code blocks, math blocks, tables
    ├─ Parent chunks: 30,000 tokens, 5,000 overlap
    └─ Child chunks: 800 tokens, no overlap
    ↓
GPT-4o-mini contextualization → OpenAI embeddings → BM25
    ↓
Qdrant → collection: academic_chunks
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
    dense:  List[float]  # 1536 dimensions, text-embedding-3-large
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
- Built-in checkpointing via Redis (same Redis already running) — enables pause/resume
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
| Storage | Qdrant (permanent) | MongoDB QAChunkModel (temporary) |
| Source | Original PDF | Extracted text already in MongoDB |

**Why 60,000 tokens?** Standard LLM context windows are ~128k tokens. Half is reserved for the prompt template, generated questions, and system instructions. 60k tokens is dense enough to generate multiple diverse questions while staying within context limits.

**Source for academic documents**: `enriched_markdown` (with image descriptions inlined) — not `markdown_text`. This gives the question generator awareness of visual content.

No re-processing of PDFs. Content is read directly from `ExtractedFictionModel` or `ExtractedAcademicModel` in MongoDB.

### 4.3 LangGraph Graph Structure

```
                    ┌─────────────┐
                    │  chunk_node │
                    │             │
                    │ Fetch from  │
                    │ MongoDB,    │
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

LangGraph persists graph state to Redis on every node transition using a `RedisSaver` checkpointer. Each job is identified by a unique `thread_id` (the `job_id`).

- **Pause**: `POST /qa-jobs/{id}/pause` → sets `QAJobModel.status = PAUSED`. LangGraph state already persisted in Redis.
- **Resume**: `POST /qa-jobs/{id}/resume` → calls `graph.ainvoke` with the same `thread_id`. LangGraph reads from Redis and continues from the last checkpoint.
- **Server restart**: no data loss. Redis holds the graph state. Job restarts from last checkpoint.
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

- JWT-based authentication (`python-jose` + `passlib`)
- Tokens issued on login, validated via FastAPI dependency injection
- Project-level role checked per endpoint via project membership lookup

---

## 7. Data Models

All collections in MongoDB using Beanie ODM.

### 7.1 Existing Models (Phase 1)

#### `DocumentModel` — collection: `documents`
```
true_title:      str          original filename
stored_title:    str          UUID-based stored name
file_type:       FileType     PDF | IMAGES
file_path:       str          absolute path on disk
data_catgory:    Datatype     FICTION | ACADEMIC
project_id:      ObjectId
```

#### `ProjectModel` — collection: `Project`
```
project_title:        str
project_description:  str
main_data_type:       Datatype    FICTION | ACADEMIC
```

#### `ChunkModel` — collection: `chunks`
```
document_id:         ObjectId
project_id:          ObjectId
chunk_index:         int
qdrant_point_id:     str         UUID reference to Qdrant point
processing_status:   str         pending | processing | completed | failed
created_at:          datetime
completed_at:        Optional[datetime]
error_message:       Optional[str]
metadata:            Optional[dict]
```

#### `ExtractedFictionModel` — collection: `extracted_fiction`
```
document_id:          ObjectId
project_id:           ObjectId
extracted_text:       str
character_count:      int
extraction_metadata:  dict       page_count, extraction_method, file_size
created_at:           datetime
```

#### `ExtractedAcademicModel` — collection: `extracted_academic`
```
document_id:           ObjectId
project_id:            ObjectId
markdown_text:         str        original Marker output
enriched_markdown:     str        with image descriptions inlined
images:                List[ExtractedImageMetadata]
  └─ filename, file_path, description, position_in_markdown, alt_text
character_count:       int
image_count:           int
extraction_metadata:   dict       marker config, conversion time, file size
created_at:            datetime
```

### 7.2 New Models (Phase 2)

#### `ProviderCredentialModel` — collection: `provider_credentials`
```
provider:          ModelProvider    openai | anthropic | google | groq | mistral |
                                    cohere | together | openrouter | azure_openai |
                                    voyageai | jina | ollama
encrypted_fields:  dict[str, str]   field_name → Fernet-encrypted value
                                    (e.g. api_key, endpoint, api_version)
updated_at:        datetime
```
> Keys are write-only from the API. Only field names are returned in status responses, never values.

#### `ProjectModelConfigModel` — collection: `project_model_configs`
```
project_id:        ObjectId
stages:            dict[ModelStage, StageModelConfig]
                     StageModelConfig: { provider, model_name, base_url? }
                     Stages: question_generator | answer_generator | validator |
                             meta_agent | embedder | reranker
embedding_locked:  bool     True once first document is processed — prevents
                            provider/model change that would corrupt Qdrant vectors
created_at:        datetime
updated_at:        datetime
```

#### `UserModel` — collection: `users`
```
email:            str         unique, indexed
hashed_password:  str
app_role:         AppRole     ADMIN | USER
created_at:       datetime
```

#### `ProjectMemberModel` — collection: `project_members`
```
project_id:    ObjectId
user_id:       ObjectId
project_role:  ProjectRole    OWNER | WORKER
added_by:      ObjectId
added_at:      datetime
```

#### `SkillConfigModel` — collection: `skill_configs`
```
project_id:                  ObjectId
version:                     int         increments on each save
question_generator_prompt:   str
answer_generator_prompt:     str
validator_prompt:            str
base_context:                str         auto-built: project type + doc summaries
questions_per_chunk:         int
question_types:              List[QuestionType]   FACTUAL | INFERENTIAL | ANALYTICAL | COT
approved:                    bool
approved_by:                 Optional[ObjectId]
approved_at:                 Optional[datetime]
created_at:                  datetime
```

#### `QAJobModel` — collection: `qa_jobs`
```
project_id:            ObjectId
skill_config_id:       ObjectId
skill_config_version:  int
status:                QAJobStatus
                         CREATED | CHUNKING | RUNNING |
                         PAUSED | COMPLETED | FAILED
created_by:            ObjectId
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

#### `QAChunkModel` — collection: `qa_chunks`
```
job_id:         ObjectId
project_id:     ObjectId
document_id:    ObjectId
chunk_index:    int
content:        str           60,000 token chunk from extracted content
token_count:    int
status:         QAChunkStatus    PENDING | PROCESSING | COMPLETED | FAILED
source_type:    Datatype         FICTION | ACADEMIC
```
> Ephemeral — safe to delete after job completes.

#### `QAPairModel` — collection: `qa_pairs`
```
job_id:                   ObjectId
project_id:               ObjectId
document_id:              ObjectId
qa_chunk_id:              ObjectId

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
reviewed_by:              Optional[ObjectId]
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
    FICTION = "FICTION"
    ACADEMIC = "ACADEMIC"

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

### 8.1 Authentication
```
POST   /auth/register                       Register new user
POST   /auth/login                          Returns JWT token
POST   /auth/logout
GET    /auth/me                             Current user info
```

### 8.2 File Management (existing)
```
POST   /files/upload                        Upload single PDF
POST   /files/upload-multiple               Upload multiple PDFs
GET    /files/{document_id}                 Get document metadata
GET    /files/                              List all documents
GET    /files/project/{project_id}          List documents in project
DELETE /files/{document_id}                 Delete document + physical file
```

### 8.3 Project Management (existing, extended)
```
POST   /projects/Create-project             Create project
GET    /projects/get-project/{id}
GET    /projects/get-all-projects
PUT    /projects/update-project/{id}
DELETE /projects/delete-project/{id}

POST   /projects/{id}/members              Add worker to project (owner only)
DELETE /projects/{id}/members/{user_id}    Remove worker
GET    /projects/{id}/members              List project members
```

### 8.4 Document Processing (existing)
```
POST   /processing/start                   Submit processing job (Celery)
GET    /processing/status/{task_id}        Check Celery task status
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

### 8.9 Credentials (admin only)
```
GET    /credentials/schema              Field definitions per provider (drives UI forms)
GET    /credentials/                    Status of all providers — configured true/false,
                                        field names present, updated_at. No key values.
POST   /credentials/{provider}          Set/update credentials
                                        Body: { fields: { api_key: "...", ... } }
DELETE /credentials/{provider}          Remove credentials
```

### 8.10 Model Configuration
```
POST   /model-config/{project_id}       Set provider + model for each stage
                                        Body: { stages: { question_generator: { provider, model_name },
                                                          embedder: { provider, model_name }, ... } }
GET    /model-config/{project_id}       Get current config
POST   /model-config/{project_id}/validate  Ping all configured stages live and return
                                            per-stage pass/fail results
```

### 8.11 Internal (worker-facing, existing)
```
GET    /internal/files/{id}/metadata
GET    /internal/files/{id}/base64
GET    /internal/files/{id}/stream
POST   /internal/extracted/fiction
POST   /internal/extracted/academic/images/{document_id}
POST   /internal/extracted/academic
```

### 8.12 Webhooks (existing)
```
POST   /webhooks/processing-complete       Called by Celery workers on completion
```

---

## 9. Infrastructure & Tech Stack

### 9.1 Full Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Web framework | FastAPI | HTTP API, background tasks |
| Async runtime | asyncio + motor | Non-blocking MongoDB access |
| ODM | Beanie | MongoDB document models |
| Database | MongoDB 7 | All persistent storage |
| Task queue | Celery | Document processing (Phase 1) |
| Message broker | Redis 7 | Celery broker |
| QA orchestration | LangGraph | Agentic QA pipeline (Phase 2) |
| Graph checkpointing | Redis (LangGraph RedisSaver) | Pause/resume state |
| Vector database | Qdrant | Dense + sparse vector search |
| PDF text extraction | PyMuPDF | Fiction documents |
| PDF → Markdown | Marker CLI | Academic documents |
| Chunking | Chonkie | Both pipelines + QA chunks |
| LLM | OpenAI · Anthropic · Google · Groq · Mistral · Cohere · Together · OpenRouter · Azure OpenAI · Ollama | All LLM stages, configurable per project |
| Embeddings | OpenAI · Google · Cohere · VoyageAI · Jina · Ollama | Dense vector generation, configurable per project |
| Vision / Marker LLM | Google Gemini | Image captioning, Marker backend |
| Reranker | Cohere · Jina · Ollama | Retrieved chunk reranking, configurable per project |
| Credential encryption | cryptography (Fernet / AES-128) | Encrypt provider API keys at rest |
| Authentication | python-jose + passlib | JWT tokens, password hashing |
| HTTP client | httpx | Async HTTP calls |
| Validation | Pydantic v2 | Request/response schemas |
| Monitoring | Flower (port 5555) | Celery task dashboard |
| Language | Python 3.11+ | |

### 9.2 Docker Compose Services

```yaml
services:
  redis:        # port 6379 — Celery broker + LangGraph checkpointer
  qdrant:       # port 6333 — vector database
  flower:       # port 5555 — Celery monitoring
  # MongoDB runs separately (Atlas or local)
```

### 9.3 Directory Structure (planned extension)

```
d:\Synthetic_Data_Genration\
│
├── web_api\
│   ├── main.py
│   ├── database.py
│   ├── data_models\
│   │   ├── BasicBeanieModels.py          DocumentModel, ProjectModel, ChunkModel
│   │   ├── ExtractedModels.py            ExtractedFictionModel, ExtractedAcademicModel
│   │   ├── CredentialModels.py           ProviderCredentialModel, schemas
│   │   ├── ModelConfigModels.py          ProjectModelConfigModel, StageModelConfig
│   │   ├── UserModels.py                 UserModel, ProjectMemberModel        [NEW]
│   │   ├── SkillModels.py                SkillConfigModel                     [NEW]
│   │   ├── QAModels.py                   QAJobModel, QAChunkModel, QAPairModel [NEW]
│   │   ├── DataModels.py                 Pydantic request/response schemas
│   │   └── enums.py                      All enums (incl. ModelProvider, ModelStage)
│   ├── routers\
│   │   ├── FileMangerRouter.py
│   │   ├── ProjectMangerRouter.py
│   │   ├── ProcessingRouter.py
│   │   ├── WebhookRouter.py
│   │   ├── InternalRouter.py
│   │   ├── CredentialRouter.py           /credentials endpoints
│   │   ├── ModelConfigRouter.py          /model-config endpoints
│   │   ├── AuthRouter.py                                                      [NEW]
│   │   ├── SkillRouter.py                                                     [NEW]
│   │   ├── QAJobRouter.py                                                     [NEW]
│   │   ├── HumanReviewRouter.py                                               [NEW]
│   │   └── DatasetRouter.py                                                   [NEW]
│   ├── services\
│   │   ├── FileHandlerService.py
│   │   ├── ProjectHandlerService.py
│   │   ├── encryption_service.py         Fernet encrypt/decrypt
│   │   ├── credential_service.py         CRUD for provider credentials
│   │   ├── llm_factory.py                get_llm_client, call_chat, call_embed, call_rerank
│   │   ├── ModelConfigService.py         per-project stage config + validation
│   │   ├── AuthService.py                                                     [NEW]
│   │   ├── SkillService.py               meta-agent prompt drafting           [NEW]
│   │   └── DatasetExportService.py       CHAT / DPO / COT export             [NEW]
│   └── qa_pipeline\                                                           [NEW]
│       ├── graph.py                      LangGraph graph definition
│       ├── nodes\
│       │   ├── chunk_node.py
│       │   ├── question_gen_node.py
│       │   ├── answer_gen_node.py
│       │   └── validate_node.py
│       ├── edges.py                      Conditional routing logic
│       └── state.py                      LangGraph state schema
│
├── workers\
│   ├── celery_app.py
│   ├── config.py
│   ├── models.py
│   ├── enums.py
│   ├── tasks\
│   │   ├── orchestrator.py
│   │   ├── fiction_processor.py
│   │   └── academic_processor.py
│   └── services\
│       ├── file_fetcher.py
│       ├── text_extractor.py
│       ├── pdf_to_markdown.py
│       ├── vision_service.py
│       ├── chunking_service_fiction.py
│       ├── chunking_service_academic.py
│       ├── contextualizer.py
│       ├── embedding_service.py
│       ├── bm25_service.py
│       ├── storage_service.py
│       ├── extracted_storage_service.py
│       └── utils\
│           ├── temp_file_manager.py
│           └── webhook_notifier.py
│
├── docker-compose.yml
├── pyproject.toml
├── requirements-worker.txt
└── .env
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
