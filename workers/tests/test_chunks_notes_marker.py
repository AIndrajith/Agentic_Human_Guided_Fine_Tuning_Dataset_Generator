"""Bugs 6, 7, 11 without network: chunking, context notes (LLM faked), Marker run settings (Marker faked)."""
import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import pytest

from workers.config import Config
from workers.models import ChildChunk, ContextChunk, ModelEndpoint
from workers.services import contextualizer as ctx_module
from workers.services import pdf_to_markdown as marker_module
from workers.services.chunk_assignment import assign_children_to_parents
from workers.services.chunking_service_fiction import ChunkingService
from workers.services.contextualizer import Contextualizer, plan_context_budget

STORY = " ".join(
    f"Chapter {i}. The knight number {i} rode through the dark forest towards the castle. "
    f"He met a dragon near the river and they talked for a long time about the old king."
    for i in range(400)
)


# ---------- bug 7: children cut once, each with exactly one parent ----------

def test_children_are_cut_once_over_the_whole_text():
    service = ChunkingService(parent_chunk_size=600, parent_overlap=150, child_chunk_size=80)
    parents, children = service.create_hierarchical_chunks(STORY)

    assert len(parents) > 3   # overlapping parents exist
    # no duplicates, no gaps in order: consecutive children don't overlap (no child overlap)
    for a, b in zip(children, children[1:]):
        assert a.end_index <= b.start_index
    # positions are counted from the start of the whole text
    for child in children:
        assert STORY[child.start_index:child.end_index].strip() == child.original_text.strip()
    # every child belongs to exactly one parent, which contains it
    owned = sorted(i for p in parents for i in p.child_indices)
    assert owned == [c.index for c in children]
    by_id = {p.context_id: p for p in parents}
    for child in children:
        parent = by_id[child.parent_context_id]
        assert parent.start_index <= child.start_index and child.end_index <= parent.end_index


def test_academic_children_are_cut_once_and_keep_tables_and_math():
    from workers.services.chunking_service_academic import AcademicChunkingService

    section = (
        "## Method\n\nWe trained the model on a large corpus of scientific papers and measured accuracy. " * 12
        + "\n\n$$E = mc^2 + \\sum_i x_i$$\n\n"
        + "| Model | Accuracy |\n|---|---|\n| A | 0.91 |\n| B | 0.87 |\n\n"
    )
    markdown = "# Paper\n\n" + section * 15
    service = AcademicChunkingService(parent_chunk_size=900, parent_overlap=200, child_chunk_size=120)
    parents, children = service.create_hierarchical_chunks(markdown)

    assert len(parents) > 2
    owned = sorted(i for p in parents for i in p.child_indices)
    assert owned == [c.index for c in children]          # each child exactly once
    spans = [(c.start_index, c.end_index) for c in children]
    assert len(spans) == len(set(spans))                  # no chunk made twice from a parent overlap
    math = [c for c in children if c.original_text.startswith("$$")]
    assert len(math) == 15 and all(c.original_text.endswith("$$") for c in math)   # math kept whole, once each


def test_child_goes_to_the_parent_where_it_is_most_central():
    parents = [
        ContextChunk(context_id=0, text="", token_count=0, start_index=0, end_index=100),
        ContextChunk(context_id=1, text="", token_count=0, start_index=70, end_index=170),
    ]
    # inside both parents; its centre (95) is 45 from parent 0's centre (50) but 25 from parent 1's (120)
    child = ChildChunk(index=0, parent_context_id=-1, original_text="x", start_index=90, end_index=100, token_count=1)
    assign_children_to_parents(parents, [child])
    assert child.parent_context_id == 1
    assert parents[1].child_indices == [0] and parents[0].child_indices == []


# ---------- bug 6: context budget ----------

def test_context_budget_follows_the_model():
    assert plan_context_budget(128_000) == ctx_module.ContextBudget(30_000, 5_000, 30)
    small = plan_context_budget(8_192)
    assert small.batch_size < 30 and small.parent_tokens >= 4 * Config.CHILD_CHUNK_SIZE
    assert plan_context_budget(None).parent_tokens == Config.PARENT_CHUNK_SIZE
    with pytest.raises(ValueError, match="too small"):
        plan_context_budget(2_000)


def test_ollama_is_told_how_much_context_to_use():
    budget = plan_context_budget(131_072)
    ollama = Contextualizer(ModelEndpoint(model="ollama_chat/llama3.1:8b", context_window=131_072), budget)
    assert budget.call_tokens <= ollama.extra_params["num_ctx"] <= 131_072   # enough for parent + batch, not max
    assert Contextualizer(ModelEndpoint(model="openai/gpt-4o-mini"), budget).extra_params == {}


# ---------- bug 6: context notes ----------

class _FakeLLM:
    """Stands in for litellm.acompletion. `answers` is a list of note-count overrides per call (None = right count)."""

    def __init__(self, answers=(), fail_always=False):
        self.answers = list(answers)
        self.fail_always = fail_always
        self.prompts: list[str] = []

    async def __call__(self, model, messages, **kwargs):
        prompt = messages[1]["content"]
        self.prompts.append(prompt)
        if self.fail_always:
            raise ValueError("model returned garbage")
        n_chunks = prompt.count("\nChunk ") + prompt.startswith("Chunk ")
        n = self.answers.pop(0) if self.answers else None
        notes = [f"note {i}" for i in range(n if n is not None else n_chunks)]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(
            {"contextual_chunks": notes})))])


def _parent_with_children(n_children: int, parent_text: str):
    parent = ContextChunk(context_id=0, text=parent_text, token_count=1, start_index=0, end_index=len(parent_text))
    children = [ChildChunk(index=i, parent_context_id=-1, original_text=f"child {i}", start_index=i, end_index=i + 1,
                           token_count=1) for i in range(n_children)]
    assign_children_to_parents([parent], children)
    return [parent], children


@pytest.fixture
def no_sleep(monkeypatch):
    async def instant(_):
        pass
    monkeypatch.setattr(ctx_module.asyncio, "sleep", instant)


async def test_whole_parent_is_sent_and_every_child_gets_a_note(monkeypatch, no_sleep):
    fake = _FakeLLM()
    monkeypatch.setattr(ctx_module.litellm, "acompletion", fake)
    long_parent = "START " + "x" * 50_000 + " END"
    parents, children = _parent_with_children(5, long_parent)

    budget = ctx_module.ContextBudget(parent_tokens=30_000, parent_overlap=5_000, batch_size=2)
    notes = await Contextualizer(ModelEndpoint(model="openai/gpt-4o-mini"), budget).contextualize_hierarchical(
        parents, children, {"title": "Book"})

    assert [n.index for n in notes] == [0, 1, 2, 3, 4]
    assert all("START" in p and "END" in p for p in fake.prompts)   # nothing cut off
    assert len(fake.prompts) == 3                                    # batches of 2, 2, 1


async def test_wrong_note_count_is_retried_not_padded(monkeypatch, no_sleep):
    fake = _FakeLLM(answers=[1])   # first answer: 1 note for 3 chunks
    monkeypatch.setattr(ctx_module.litellm, "acompletion", fake)
    parents, children = _parent_with_children(3, "parent")

    notes = await Contextualizer(ModelEndpoint(model="openai/gpt-4o-mini")).contextualize_hierarchical(
        parents, children, {})
    assert len(fake.prompts) == 2 and [n.context_description for n in notes] == ["note 0", "note 1", "note 2"]
    assert "Context not available" not in str(notes)


async def test_a_failed_batch_fails_the_document(monkeypatch, no_sleep):
    monkeypatch.setattr(ctx_module.litellm, "acompletion", _FakeLLM(fail_always=True))
    parents, children = _parent_with_children(3, "parent")
    with pytest.raises(RuntimeError, match="Context notes failed for 1 of 1 batches"):
        await Contextualizer(ModelEndpoint(model="openai/gpt-4o-mini")).contextualize_hierarchical(
            parents, children, {})


# ---------- bug 11: Marker settings ----------

def test_marker_uses_the_projects_vision_model(monkeypatch):
    service = marker_module.PDFToMarkdownService()
    config, env = service.build_run(Path("paper.pdf"), Path("out"),
                                    ModelEndpoint(model="gemini/gemini-2.5-flash", api_key="g-key"))
    assert config["llm_service"] == "marker.services.gemini.GoogleGeminiService"
    assert config["gemini_model_name"] == "gemini-2.5-flash" and config["gemini_api_key"] == "g-key"
    assert config["force_ocr"] is False and config["use_llm"] is True
    assert env == {"GOOGLE_API_KEY": "g-key"}

    config, _ = service.build_run(Path("p.pdf"), Path("o"), ModelEndpoint(
        model="azure/my-deploy", api_key="a", api_base="https://x.openai.azure.com", api_version="2024-10-21"))
    assert config["llm_service"].endswith("AzureOpenAIService") and config["deployment_name"] == "my-deploy"

    # a provider Marker can't use and no .env key -> Marker runs without an LLM
    monkeypatch.setattr(marker_module.Config, "GEMINI_API_KEY", None)
    config, env = service.build_run(Path("p.pdf"), Path("o"), ModelEndpoint(model="groq/llava", api_key="k"))
    assert "use_llm" not in config and "llm_service" not in config and env == {}


async def test_marker_keys_never_reach_the_command_line(monkeypatch):
    seen = {}

    class _Proc:
        pid, returncode = 1234, 0

        async def communicate(self):
            return b"", b""

    async def fake_exec(*args, env=None, **kwargs):
        seen["args"], seen["env"] = args, env
        return _Proc()

    monkeypatch.setattr(marker_module.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setenv("OPENAI_API_KEY", "leaky-parent-key")
    service = marker_module.PDFToMarkdownService()
    config, env = service.build_run(Path("paper.pdf"), Path("out"),
                                    ModelEndpoint(model="anthropic/claude-sonnet-4-5", api_key="secret-key"))
    await service._run_marker(config, env)

    assert "secret-key" not in " ".join(map(str, seen["args"]))
    assert json.loads(seen["env"]["MARKER_RUNNER_CONFIG"])["claude_api_key"] == "secret-key"
    assert "OPENAI_API_KEY" not in seen["env"]   # unrelated keys from the worker's env are not passed on
