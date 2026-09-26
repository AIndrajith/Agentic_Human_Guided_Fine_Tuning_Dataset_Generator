# API Guide

A plain-language guide to the `web_api`: what each endpoint does, what to send, what comes back, and what happens behind it. Field names are exact — copy them as written.

> The live contract is always at **http://localhost:8000/docs** (Swagger). This guide explains the *why* and the *flow*; Swagger shows the exact schema and lets you try calls.

**Contents**
1. [The big picture](#1-the-big-picture)
2. [Basics you need for every call](#2-basics-you-need-for-every-call)
3. [Who can do what](#3-who-can-do-what)
4. [A full walkthrough, start to finish](#4-a-full-walkthrough-start-to-finish)
5. [Endpoint reference](#5-endpoint-reference)
6. [How processing works (API ↔ worker)](#6-how-processing-works-api--worker)
7. [Statuses](#7-statuses)
8. [Where data is stored](#8-where-data-is-stored)
9. [Error codes](#9-error-codes)
10. [Where the code lives](#10-where-the-code-lives)
11. [Known gaps](#11-known-gaps)

---

## 1. The big picture

```
 You / front-end ──HTTP──►  web_api (FastAPI)  ──►  PostgreSQL   users, projects, documents, jobs...
                               │    ▲           ──►  MinIO        the actual files + extracted text
                     puts a    │    │ worker calls back
                     task on   ▼    │ (/internal, /webhooks)
                             Redis ──► Celery worker ──► Qdrant (vectors for search)
```

- **web_api** is the only thing that touches PostgreSQL and MinIO.
- **Workers** do the heavy work (read PDFs, chunk, embed). They never touch the database directly — they *ask* the API for files and *tell* the API when they're done, using a shared secret (`X-Internal-Token`).
- **Redis** is just the queue between the API and the workers.

---

## 2. Basics you need for every call

**Base URL:** `http://localhost:8000`

**Logging in.** Call `POST /users/login`, get an `access_token`, then send it on every other call:
```
Authorization: Bearer <access_token>
```
The token lasts **20 minutes** (`JWT_EXPIRE_MINUTES`). There is no refresh yet — log in again when it expires.
In Swagger, click **Authorize** and put your **email** in the `username` box.

**IDs** are UUIDs, e.g. `3f2b9c1e-8a4d-4f6e-9b2a-1c7d5e0f4a21`. A badly formed ID in the URL gives a `422` automatically.

**Errors** (from our code) always look like this:
```json
{ "error": { "code": "project_not_found", "message": "Project not found." } }
```
Two exceptions come from FastAPI itself, in its own format:
- request **shape** is wrong (missing field, wrong type) → `422` with `{ "detail": [ ... ] }`
- **no token at all** → `401` with `{ "detail": "Not authenticated" }`

**Status codes you'll see**

| Code | Meaning |
|---|---|
| 200 / 201 / 202 | OK / created / accepted (work queued) |
| 204 | OK, nothing to return (deletes) |
| 400 | Bad request (e.g. unsupported file type) |
| 401 | Not logged in, bad/expired token, or bad internal token |
| 403 | Logged in, but not allowed |
| 404 | Not found — **also** returned when you're not a member of the project (so project IDs can't be guessed) |
| 409 | Conflicts with current state (duplicate, locked, in use) |
| 413 | File too large |
| 422 | Input breaks a rule (weak password, wrong provider for a stage...) |

---

## 3. Who can do what

There are two kinds of roles:

- **App role** (on the user): `admin` or `user`.
- **Project role** (per project): `owner` or `worker`. Whoever creates a project becomes its `owner`.

**Admins can do everything in every project**, even without being a member.

| Action | Who |
|---|---|
| Invite users, list users, (de)activate users | admin |
| Manage credentials (API keys) | admin |
| See credential names (to pick one) | anyone logged in |
| Create a project | anyone logged in |
| View a project, its members, documents, model config, jobs, extractions | project member |
| Upload / delete documents, start processing | project member (owner or worker) |
| Edit / delete project, manage members, change model config | project owner |

---

## 4. A full walkthrough, start to finish

This is the whole happy path. Each step links to its reference section.

**Step 0 — create the first admin** (one time, from the terminal; the API can't invite without an admin):
```powershell
uv run --project web_api python -m web_api.scripts.create_admin --email you@example.com --username admin
```

**Step 1 — admin logs in** → [`POST /users/login`](#post-userslogin)

**Step 2 — admin invites a user** → [`POST /users`](#post-users)
The user gets an email with a link: `{HOST_URL}/setup-password?token=...`
(If email isn't configured, the link is printed in the API log instead.)

**Step 3 — user sets username + password** → [`POST /users/setup`](#post-userssetup) with the token from the link. The link works **once**, for **24 hours**.

**Step 4 — user logs in** → `POST /users/login`

**Step 5 — create a project** → [`POST /projects`](#post-projects). Pick `data_type`: `fiction` or `academic`. It decides how documents are processed.

**Step 6 — upload PDFs** → [`POST /projects/{project_id}/documents`](#post-projectsproject_iddocuments)

**Step 7 — admin adds an API key** → [`POST /credentials`](#post-credentials) (e.g. an OpenAI key, named "OpenAI team key")

**Step 8 — owner chooses models for the project** → [`PUT /projects/{project_id}/model-config`](#put-projectsproject_idmodel-config). At minimum set `meta_agent` (used to write chunk context) and `embedder`. Optional: test first with `/validate`.

**Step 9 — start processing** → [`POST /projects/{project_id}/processing-jobs`](#post-projectsproject_idprocessing-jobs). You get a job back with status `queued`.

**Step 10 — watch progress** → [`GET /processing-jobs/{job_id}`](#get-processing-jobsjob_id). Documents move `queued → processing → completed/failed`.

**Step 11 — read the extracted text** → [`GET /documents/{document_id}/extraction`](#get-documentsdocument_idextraction)

---

## 5. Endpoint reference

Format for each: **who can call** · what to send · what you get · things to know.

### 5.1 Users & login

Most user endpoints return a **User** object:
```json
{
  "id": "uuid",
  "email": "bob@example.com",
  "username": "bob",              // null until the user finishes setup
  "app_role": "user",             // "admin" | "user"
  "is_active": true,
  "must_change_password": false,  // true = invited but hasn't set a password yet
  "created_at": "2026-09-26T10:00:00Z"
}
```

#### `POST /users`
**Admin.** Invite someone. Creates the user (no password yet) and emails a setup link.
```json
{ "email": "bob@example.com", "app_role": "user" }   // app_role optional, default "user"
```
→ `201` User. Email is stored lowercase.
Errors: `409 user_already_exists`.

#### `GET /users`
**Admin.** All users → list of User.

#### `POST /users/resend-invite`
**Admin.** Sends a **new** link; the old link stops working.
```json
{ "email": "bob@example.com" }
```
→ User. Errors: `404 user_not_found`, `409` if the user already finished setup.

#### `PATCH /users/{user_id}/active`
**Admin.** Turn an account off or on.
```json
{ "is_active": false }
```
→ User. Takes effect **immediately** — even tokens already issued stop working.
Errors: `409` if you try to deactivate yourself.

#### `POST /users/setup`
**Public** (no token). The invited user finishes their account.
```json
{ "token": "<from the email link>", "username": "bob", "password": "Str0ng!Pass" }
```
- `username`: 3–64 chars, letters/numbers/`_ . -`
- `password`: 8+ chars, with an uppercase, a lowercase, a number, and a symbol

→ User (with `must_change_password: false`).
Errors: `401 invalid_token` (wrong, used, or expired link), `403 account_not_active`, `409` username taken / already set up, `422 validation_error` (weak password — the message lists what's missing).

#### `POST /users/login`
**Public.** Sent as a **form**, not JSON (this is the OAuth2 standard):
```
username=bob@example.com      ← yes, the field is called "username" but takes the EMAIL
password=Str0ng!Pass
```
→ `{ "access_token": "eyJ...", "token_type": "bearer" }`
Errors: `401 invalid_credentials` (same error for unknown email and wrong password — on purpose), `403 account_not_active`.

#### `GET /users/me`
**Logged in.** → your User.

---

### 5.2 Projects

**Project** object:
```json
{
  "id": "uuid",
  "title": "Sherlock Holmes",
  "description": "",
  "data_type": "fiction",        // "fiction" | "academic"
  "created_by": "uuid",
  "embedding_locked": false,     // becomes true after the first document is processed
  "created_at": "...", "updated_at": "..."
}
```

#### `POST /projects`
**Logged in.** You become the owner.
```json
{ "title": "Sherlock Holmes", "description": "optional", "data_type": "fiction" }
```
→ `201` Project.

#### `GET /projects`
**Logged in.** Projects you're a member of (admins: all), newest first.

#### `GET /projects/{project_id}`
**Member.** → Project.

#### `PATCH /projects/{project_id}`
**Owner.** Send only what you want to change.
```json
{ "title": "New title" }
```
→ Project. Errors: `409` if you change `data_type` after documents were uploaded.

#### `DELETE /projects/{project_id}`
**Owner.** → `204`. Deletes **everything** of the project: members, documents, model config, jobs, chunks (database cascades), and all its files in MinIO.

---

### 5.3 Project members

**Member** object:
```json
{ "user_id": "uuid", "email": "bob@example.com", "username": "bob",
  "role": "worker", "added_by": "uuid", "added_at": "..." }
```

| Endpoint | Who | Send | Notes |
|---|---|---|---|
| `GET /projects/{project_id}/members` | member | — | list of Member |
| `POST /projects/{project_id}/members` | owner | `{ "email": "...", "role": "worker" }` | `201`. The user must already exist (invite first). `404 user_not_found`, `409` already a member |
| `PATCH /projects/{project_id}/members/{user_id}` | owner | `{ "role": "owner" }` | `409` if it would leave zero owners |
| `DELETE /projects/{project_id}/members/{user_id}` | owner | — | `204`. `409` if it's the last owner |

---

### 5.4 Documents

**Document** object:
```json
{
  "id": "uuid",
  "project_id": "uuid",
  "original_name": "book.pdf",
  "file_type": "pdf",              // "pdf" | "images"
  "content_type": "application/pdf",
  "size_bytes": 2712052,
  "status": "uploaded",            // see Statuses
  "error": null,                   // filled when processing failed
  "uploaded_by": "uuid",
  "created_at": "...",
  "processed_at": null
}
```

#### `POST /projects/{project_id}/documents`
**Member.** Upload one or many files as `multipart/form-data`, field name **`files`** (repeat it for each file).
→ `201` list of Document.
- Allowed: `.pdf`, `.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`
- The file **content** is checked, not just the extension (a renamed `.exe` is rejected).
- Max size per file: `MAX_UPLOAD_MB` (default 200 MB).
- **All-or-nothing:** if one file fails, none are saved.

Errors: `400 unsupported_file_type`, `413 file_too_large`, `422` (no files / empty file).

#### `GET /projects/{project_id}/documents`
**Member.** List, oldest first.

#### `GET /documents/{document_id}` · `DELETE /documents/{document_id}`
**Member.** Get one / delete one (`204`, also removes the file from MinIO).

---

### 5.5 Credentials (API keys)

Credentials are **global**: an admin adds a key once (e.g. "OpenAI team key"), and any project can use it. Keys are encrypted in the database and **never** returned by the API.

**Credential** object (admin view):
```json
{
  "id": "uuid",
  "name": "OpenAI team key",
  "provider": "openai",
  "fields_present": ["api_key"],     // names only, never values
  "used_by_projects": [ { "project_id": "uuid", "title": "Sherlock Holmes" } ],
  "created_by": "uuid", "created_at": "...", "updated_at": "..."
}
```

**Which fields each provider needs** (`GET /credentials/schema` returns this too):

| Provider | Fields |
|---|---|
| `openai`, `anthropic`, `google`, `groq`, `mistral`, `cohere`, `together`, `openrouter`, `voyageai`, `jina` | `api_key` |
| `azure_openai` | `api_key`, `endpoint`, `api_version` |
| `ollama` | `base_url` |

| Endpoint | Who | Send / get |
|---|---|---|
| `GET /credentials/schema` | logged in | `[{ "provider", "required_fields" }]` |
| `GET /credentials/options` | logged in | `[{ "id", "name", "provider" }]` — what an owner picks from |
| `GET /credentials` | admin | list of Credential |
| `GET /credentials/{credential_id}` | admin | Credential |
| <a id="post-credentials"></a>`POST /credentials` | admin | `{ "name": "OpenAI team key", "provider": "openai", "fields": { "api_key": "sk-..." } }` → `201` |
| `PATCH /credentials/{credential_id}` | admin | `{ "name"?: "...", "fields"?: {...} }` — `fields` **replaces all** keys (use it to rotate a key; projects pick it up automatically) |
| `DELETE /credentials/{credential_id}` | admin | `204` |

Errors: `422` missing or unknown fields (message says which), `409` name already used, `409 credential_in_use` on delete (message lists the projects — detach it from them first).

---

### 5.6 Model config (which model does which job)

A project has **stages** — jobs an AI model does. For each stage you attach a credential + a model name.

| Stage | What it's for | Allowed providers |
|---|---|---|
| `meta_agent` | writes the context blurb for each chunk (used now) | openai, anthropic, google, groq, mistral, cohere, together, openrouter, azure_openai, ollama |
| `embedder` | turns text into vectors (used now) | openai, google, ollama, cohere, voyageai, jina |
| `vision` | describes images in academic papers (used now) | google, openai |
| `question_generator`, `answer_generator`, `validator` | QA generation (planned) | same as `meta_agent` |
| `reranker` | reorders search results (planned) | cohere, jina, ollama |

> Today the workers only really support **OpenAI-compatible** providers for `meta_agent` / `embedder`. Other providers are accepted here but may fail in the worker (see [Known gaps](#11-known-gaps)).

#### `GET /projects/{project_id}/model-config`
**Member.**
```json
{
  "project_id": "uuid",
  "embedding_locked": false,
  "stages": [
    { "stage": "embedder", "credential_id": "uuid", "credential_name": "OpenAI team key",
      "provider": "openai", "model_name": "text-embedding-3-large",
      "base_url": null, "embedding_dim": 3072, "updated_at": "..." }
  ]
}
```

#### `PUT /projects/{project_id}/model-config`
**Owner.** Add or update stages. Stages you don't send are left alone.
```json
{
  "stages": {
    "meta_agent": { "credential_id": "uuid", "model_name": "gpt-4o-mini" },
    "embedder":   { "credential_id": "uuid", "model_name": "text-embedding-3-large", "embedding_dim": 3072 }
  }
}
```
- `embedding_dim` is **required** for `embedder` and **not allowed** on other stages. It must match the model (e.g. 3072 for `text-embedding-3-large`).
- `base_url` is optional — overrides the credential's URL (for Ollama / self-hosted).
- After the first document is processed, the **embedder is locked** (`embedding_locked: true`) because changing it would make old vectors useless.

→ the same object as GET.
Errors: `404 credential_not_found`, `422` provider can't do that stage / `embedding_dim` rule, `409` embedder is locked.

#### `DELETE /projects/{project_id}/model-config/{stage}`
**Owner.** Remove one stage → `204`. (`409` for a locked embedder.)

#### `POST /projects/{project_id}/model-config/validate`
**Owner.** Same body as PUT. **Saves nothing** — it makes a tiny real call to each provider to prove the key and model name work (costs a fraction of a cent).
```json
{ "all_ok": false, "results": [ { "stage": "embedder", "ok": false, "error": "model not found" } ] }
```

---

### 5.7 Processing

**Job** object:
```json
{
  "id": "uuid",
  "project_id": "uuid",
  "status": "running",             // queued | running | completed | partial | failed
  "error": null,
  "requested_by": "uuid",
  "created_at": "...", "started_at": "...", "finished_at": null,
  "documents": [
    { "document_id": "uuid", "original_name": "book.pdf", "status": "processing", "error": null }
  ]
}
```

#### `POST /projects/{project_id}/processing-jobs`
**Member.** Queue documents for the worker.
```json
{ "document_ids": ["uuid", "uuid"] }     // optional
```
- **No body / no ids** = process every document that is `uploaded` or `failed` (so re-running only retries failures).
- Only **PDFs** can be processed right now.

→ `202` Job (status `queued`).
Errors: `404 document_not_found` (id not in this project), `409` a document is already queued/processing, `422` nothing to process / not a PDF, `409` "Could not queue the job" (Redis is down — the job is saved as `failed`).

#### `GET /projects/{project_id}/processing-jobs`
**Member.** All jobs, newest first — same as Job but **without** the `documents` list.

#### `GET /processing-jobs/{job_id}`
**Member of the job's project.** → Job with per-document status. Poll this to watch progress.

#### `GET /projects/{project_id}/extractions`
**Member.** Summary of extracted content for every processed document:
```json
[ { "document_id": "uuid", "char_count": 412000, "image_count": 0,
    "metadata": { "page_count": 310, "extraction_method": "PyMuPDF" },
    "images": [], "created_at": "..." } ]
```

#### `GET /documents/{document_id}/extraction`
**Member.** The same summary fields as above, **plus the text** (loaded from MinIO):
```json
{
  "document_id": "uuid", "char_count": 18250, "image_count": 1, "metadata": {}, "created_at": "...",
  "text": "full extracted text (fiction) or original markdown (academic)",
  "enriched_text": "academic only: markdown with images replaced by their descriptions",
  "images": [ { "filename": "fig_1.png", "storage_key": "projects/.../images/fig_1.png",
                "description": "A bar chart showing...", "position": 1042 } ]
}
```
Errors: `404 extraction_not_found` (not processed yet).

---

## 6. How processing works (API ↔ worker)

What actually happens after you call `POST /projects/{project_id}/processing-jobs`:

```
 1. API     creates a processing_jobs row + one job_documents row per document
            documents.status = "queued"
 2. API     reads the project's model config, decrypts the keys, and puts a task on Redis
 3. Worker  picks up the task, then for each document:
      a. GET  /internal/files/{id}/metadata      → API marks the document "processing"
      b. GET  /internal/files/{id}/base64         (files under 5 MB)
         or   /internal/files/{id}/stream         (5 MB and up)
      c. extracts text, POSTs it to /internal/extracted/fiction   (or /academic + images)
      d. chunks, embeds, saves vectors in Qdrant
      e. POST /webhooks/processing-complete       → API saves chunk links, marks "completed"
         or   /webhooks/processing-failed         → API marks "failed" with the stage + error
 4. API     after each webhook, recalculates the job status (running / completed / partial / failed)
```

**The task the API sends** (Celery task name `workers.tasks.process_documents`):
```json
{
  "task_id": "<job id>",
  "project_id": "uuid",
  "documents": [ { "id": "uuid", "file_size": 2712052 } ],
  "data_type": "fiction",
  "credentials": {
    "llm_provider": "openai", "llm_model": "gpt-4o-mini", "llm_api_key": "sk-...", "llm_base_url": null,
    "embed_provider": "openai", "embed_model": "text-embedding-3-large", "embed_api_key": "sk-...", "embed_base_url": null,
    "vision_api_key": "...", "vision_model": "gemini-2.0-flash"
  }
}
```
Stages that aren't configured are simply left out; the worker then falls back to `OPENAI_API_KEY` / `GEMINI_API_KEY` from `.env`.

### Worker-only endpoints

All need the header **`X-Internal-Token: <INTERNAL_API_TOKEN from .env>`**. A normal user token does **not** work here. Wrong/missing → `401 invalid_internal_token`.

| Endpoint | Send | Get back |
|---|---|---|
| `GET /internal/files/{document_id}/metadata` | — | `{ document_id, filename, stored_filename, file_size, file_type, data_category, project_id, should_stream }` — `should_stream` is true at 5 MB+ |
| `GET /internal/files/{document_id}/base64` | — | metadata + `file_data` (base64). `400` if the file is 5 MB+ |
| `GET /internal/files/{document_id}/stream` | — | the raw file bytes |
| `POST /internal/extracted/fiction` | `{ document_id, project_id, extracted_text, extraction_metadata }` | extraction summary |
| `POST /internal/extracted/academic/images/{document_id}` | multipart, field **`images`** | `{ document_id, images_saved, saved_keys: [{ filename, minio_key }] }` |
| `POST /internal/extracted/academic` | `{ document_id, project_id, markdown_text, enriched_markdown, images: [{ filename, description, position_in_markdown, alt_text }], extraction_metadata }` | extraction summary |
| `POST /webhooks/processing-complete` | `{ task_id, document_id, project_id, status, chunks_processed, total_chunks, chunks_data: [{ chunk_index, qdrant_point_id, metadata }], error_message }` | `{ message, document_id, status, chunks_created }` |
| `POST /webhooks/processing-failed` | `{ task_id, document_id, error_message, stage }` | `{ message, document_id, status }` |

Notes:
- In `processing-complete`, `status: "failed"` means failed; **anything else** counts as completed.
- Sending `processing-complete` twice for the same document (a worker retry) **replaces** the old chunk rows — no duplicates.
- Image filenames are cleaned: any folder part is dropped (`../../x.png` → `x.png`) so a worker can't write outside the document's folder.

---

## 7. Statuses

**Document**
```
uploaded ──► queued ──► processing ──► completed
                │            │
                └────────────┴──────► failed ──► (can be queued again)
```

**Job** (worked out from its documents)

| Status | When |
|---|---|
| `queued` | created, worker hasn't started |
| `running` | at least one document is still queued or processing |
| `completed` | every document completed |
| `failed` | every document failed (or the task couldn't be queued) |
| `partial` | finished, some completed and some failed |

---

## 8. Where data is stored

**PostgreSQL** (tables, in `web_api/db/models/`)

| Table | Holds |
|---|---|
| `users`, `email_events` | accounts, invite emails sent |
| `projects`, `project_members` | projects and who's in them |
| `documents` | one row per uploaded file (status lives here) |
| `provider_credentials` | encrypted API keys |
| `project_stage_models` | model config: stage → credential + model |
| `processing_jobs`, `job_documents` | jobs and per-document job status |
| `extractions`, `extracted_images` | where the extracted text/images are in MinIO, plus stats |
| `chunks` | links each chunk to its Qdrant point |
| `langgraph.*` | LangGraph checkpoints (for the future QA pipeline) |

**MinIO** (bucket `MINIO_BUCKET`, default `synthetic-data`) — everything for a project sits under one folder, so deleting a project removes it all:
```
projects/{project_id}/documents/{document_id}.pdf
projects/{project_id}/extracted/{document_id}/text.txt        fiction
projects/{project_id}/extracted/{document_id}/text.md         academic, original
projects/{project_id}/extracted/{document_id}/enriched.md     academic, images described
projects/{project_id}/extracted/{document_id}/images/fig_1.png
```

**Qdrant** — the chunk vectors (collections `fiction_chunks`, `academic_chunks`), written by the worker.

---

## 9. Error codes

| `code` | HTTP | Usually means |
|---|---|---|
| `invalid_credentials` | 401 | wrong email or password |
| `invalid_token` | 401 | bad/expired login token, or used/expired setup link |
| `invalid_internal_token` | 401 | worker call without the right `X-Internal-Token` |
| `forbidden` | 403 | needs admin |
| `account_not_active` | 403 | user was deactivated |
| `project_access_denied` | 403 | you're a member but need to be owner |
| `user_not_found`, `project_not_found`, `document_not_found`, `member_not_found`, `credential_not_found`, `job_not_found`, `extraction_not_found` | 404 | doesn't exist (or you're not in that project) |
| `user_already_exists` | 409 | email already invited |
| `credential_in_use` | 409 | credential still attached to a project |
| `conflict` | 409 | other conflicts — read the `message` |
| `unsupported_file_type` | 400 | bad file type or content doesn't match the extension |
| `bad_request` | 400 | e.g. base64 requested for a big file |
| `file_too_large` | 413 | over `MAX_UPLOAD_MB` |
| `validation_error` | 422 | a rule was broken — the `message` says which |
| `internal_error` | 500 | a bug — details are in the API log |

---

## 10. Where the code lives

Every request goes **router → service → database**. Routers only deal with HTTP; services hold the logic.

| If you want to change... | Look in |
|---|---|
| An endpoint's URL, who can call it | `web_api/routers/*Router.py` |
| What an endpoint actually does | `web_api/services/*Service.py` |
| Request / response fields | `web_api/data_models/` (`DataModels.py`, `CredentialModels.py`, `ModelConfigModels.py`, `ProcessingModels.py`) |
| Allowed values (roles, statuses, stages, providers) | `web_api/data_models/enums.py` |
| Database tables | `web_api/db/models/` — then create a migration (see [development guide](development.md)) |
| Login / admin / member checks | `web_api/deps/auth.py`, `web_api/deps/projects.py` |
| Worker token check | `web_api/deps/internal.py` |
| Error codes | `web_api/errors.py` |
| Settings / env vars | `web_api/core/config.py` |
| Which provider can serve which stage | `STAGE_CAPABILITIES` in `web_api/services/llm_factory.py` |
| Tests showing each flow in action | `web_api/tests/` — readable examples of every endpoint above |

---

## 11. Known gaps

Things that are true today and worth remembering:

- **No token refresh.** Tokens expire after 20 minutes; log in again.
- **Keys travel through Redis.** Decrypted API keys are inside the Celery task message. Planned fix: the worker fetches them from the API instead.
- **Any member can delete documents** — not only owners. Change it in `FileMangerRouter.py` if you want owner-only.
- **Worker bugs** from the audit are not fixed yet (academic pipeline incomplete, embedding size hardcoded in the worker, BM25 hashing, non-OpenAI providers in the worker). The API side is ready for them.
- **Images** can be uploaded but not processed yet.
- **QA generation, review and export** endpoints (in [design.md](design.md) §8.5–8.8) don't exist yet.
