"""Worker tests without network: BM25 requests, deterministic point ids, retry rules, per-document isolation.

From the repo root:  uv run --project workers pytest workers/tests
"""
import os

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import httpx
import litellm
import pytest

from workers.models import EmbedderEndpoint, ModelEndpoint, ProcessingConfig, TaskData
from workers.services.bm25_service import BM25Service
from workers.services.storage_service import point_id
from workers.tasks import orchestrator
from workers.utils import retry as retry_module
from workers.utils.retry import is_permanent_api_error, is_transient, retry_async


# ---------- bug 3: BM25 (computed by Qdrant's built-in model) ----------

def test_bm25_documents_and_queries_use_the_same_model_and_options():
    bm25 = BM25Service()
    doc = bm25.document("The dragon guarded the castle.")
    query = bm25.query_document("dragons")
    assert doc.model == query.model == "qdrant/bm25"
    assert doc.options == query.options == {"k": 1.2, "b": 0.75, "avg_len": 400.0, "language": "english"}
    assert [d.text for d in bm25.documents(["a", "b"])] == ["a", "b"]


def test_old_qdrant_is_detected():
    from workers.services.storage_service import MIN_QDRANT_VERSION, _version_tuple
    assert _version_tuple("1.13.2") < MIN_QDRANT_VERSION <= _version_tuple("1.18.0")
    assert _version_tuple("1.15.2-dev") == MIN_QDRANT_VERSION


# ---------- bug 4: deterministic point ids ----------

def test_point_ids_are_deterministic():
    assert point_id("doc-1", 0) == point_id("doc-1", 0)
    assert len({point_id("doc-1", 0), point_id("doc-1", 1), point_id("doc-2", 0)}) == 3


# ---------- bug 5: retry rules ----------

def _status_error(code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "http://x")
    return httpx.HTTPStatusError("boom", request=request, response=httpx.Response(code, request=request))


def test_only_temporary_errors_are_retried():
    assert is_transient(httpx.ConnectError("refused"))
    assert is_transient(_status_error(503)) and is_transient(_status_error(429))
    assert not is_transient(_status_error(404))
    assert is_transient(litellm.RateLimitError("slow down", llm_provider="openai", model="m"))
    assert not is_transient(litellm.AuthenticationError("bad key", llm_provider="openai", model="m"))
    assert is_permanent_api_error(litellm.NotFoundError("no model", llm_provider="openai", model="m"))
    assert not is_transient(ValueError("broken pdf"))


async def test_retry_async_retries_then_gives_up_early_on_permanent(monkeypatch):
    async def no_sleep(_):
        pass
    monkeypatch.setattr(retry_module.asyncio, "sleep", no_sleep)

    calls = []

    async def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise httpx.ConnectError("refused")
        return "ok"

    assert await retry_async(flaky, what="flaky") == "ok" and len(calls) == 3

    calls.clear()

    async def broken():
        calls.append(1)
        raise ValueError("broken pdf")

    with pytest.raises(ValueError):
        await retry_async(broken, what="broken")
    assert len(calls) == 1


# ---------- bug 5: one failing document doesn't stop or repeat the others ----------

_CONFIG = ProcessingConfig(
    project_id="p", data_type="fiction", qdrant_collection="project_p",
    llm=ModelEndpoint(model="openai/gpt-4o-mini"),
    embedder=EmbedderEndpoint(model="openai/text-embedding-3-large", dimension=3072),
)


class _FakeProcessor:
    def __init__(self):
        self.seen = []

    async def process_document(self, task_id, document, project_id, config):
        self.seen.append(document.id)
        if document.id == "bad":
            raise ValueError("broken pdf")
        return {"status": "completed"}


def _task(*doc_ids):
    return TaskData(task_id="t", project_id="p", data_type="fiction",
                    documents=[{"id": d, "file_size": 1} for d in doc_ids])


async def test_failed_document_does_not_stop_the_others(monkeypatch):
    fake = _FakeProcessor()
    monkeypatch.setattr(orchestrator, "FictionProcessor", lambda: fake)

    async def config(_):
        return _CONFIG
    monkeypatch.setattr(orchestrator, "fetch_processing_config", config)

    result = await orchestrator._process_documents_async(_task("a", "bad", "c"))
    assert fake.seen == ["a", "bad", "c"]   # each document exactly once
    assert result["successful"] == 2 and result["failed"] == 1


async def test_web_api_unreachable_retries_the_task_not_the_documents(monkeypatch):
    fake = _FakeProcessor()
    monkeypatch.setattr(orchestrator, "FictionProcessor", lambda: fake)

    async def unreachable(_):
        raise httpx.ConnectError("refused")
    monkeypatch.setattr(orchestrator, "fetch_processing_config", unreachable)

    with pytest.raises(orchestrator._ConfigUnavailable):
        await orchestrator._process_documents_async(_task("a"))
    assert fake.seen == []
