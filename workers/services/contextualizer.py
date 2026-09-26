"""
Service for contextualizing chunks using LLM.

Each child chunk gets a short context note written from its WHOLE parent. The parent goes first in the
prompt and is identical for every batch of that parent, so providers with prompt caching (OpenAI, Gemini,
DeepSeek, ...) can reuse it for the next batches. If any batch still fails after its retries, the whole
document fails: a chunk is never stored without its note.
"""

import asyncio
from dataclasses import dataclass
from typing import List, Optional
import litellm
from workers.models import (
 ContextualOutput,ContextChunk, ChildChunk, ContextualizedChildChunk, ModelEndpoint
)
from workers.config import Config
from workers.utils.retry import is_permanent_api_error
import logging

logger = logging.getLogger(__name__)

litellm.drop_params = True   # drop params a provider doesn't support (e.g. temperature on some models)

SYSTEM_PROMPT = f"""You write short context notes for chunks of a document, so each chunk can be found by search on its own.

You get the document title, a large <parent_context> section and several numbered <child_chunks> taken from it.
For each child chunk, write one context description ({Config.CONTEXT_DESCRIPTION_MIN_TOKENS}-{Config.CONTEXT_DESCRIPTION_MAX_TOKENS} tokens) that:
- identifies what part of the narrative/content this is
- mentions key themes, characters, or concepts present
- relates it to the parent context

Return ONLY the descriptions, NOT the chunk text. Answer only with a JSON object whose "contextual_chunks"
array has exactly one description per chunk, in the same order, like:
{{"contextual_chunks": ["description of chunk 1", "description of chunk 2"]}}"""

PROMPT_OVERHEAD_TOKENS = 1500   # system prompt + title + tags
CONTEXT_WINDOW_SAFETY = 0.85    # models count tokens differently than our tokenizer; keep a margin


def parse_contextual_output(content: str) -> ContextualOutput:
    """Validate the model's JSON answer. Tolerates ```json fences and text around the object."""
    text = (content or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        raise ValueError(f"No JSON object in model output: {text[:200]!r}")
    return ContextualOutput.model_validate_json(text[start:end + 1])


@dataclass(frozen=True)
class ContextBudget:
    """How big parents and batches can be so one call fits the model's context window."""
    parent_tokens: int
    parent_overlap: int
    batch_size: int
    child_tokens: int = Config.CHILD_CHUNK_SIZE

    @property
    def call_tokens(self) -> int:
        """Context one call needs (parent + batch in, notes out), with the tokenizer margin."""
        per_child = self.child_tokens + Config.CONTEXT_DESCRIPTION_MAX_TOKENS
        needed = self.parent_tokens + self.batch_size * per_child + PROMPT_OVERHEAD_TOKENS
        return int(needed / CONTEXT_WINDOW_SAFETY)


def plan_context_budget(context_window: Optional[int], child_tokens: int = Config.CHILD_CHUNK_SIZE) -> ContextBudget:
    parent, overlap, batch = Config.PARENT_CHUNK_SIZE, Config.PARENT_CHUNK_OVERLAP, Config.LLM_MAX_CHUNKS_PER_BATCH
    if not context_window:
        logger.warning("Context window of the meta_agent model is unknown; using default chunk sizes")
        return ContextBudget(parent, overlap, batch, child_tokens)

    usable = int(context_window * CONTEXT_WINDOW_SAFETY) - PROMPT_OVERHEAD_TOKENS
    per_child = child_tokens + Config.CONTEXT_DESCRIPTION_MAX_TOKENS   # the chunk in + its note out
    min_parent = 4 * child_tokens

    if usable - batch * per_child < min_parent:   # small model: fewer children per call
        batch = max(1, (usable - min_parent) // per_child)
    room = usable - batch * per_child
    if room < child_tokens:
        raise ValueError(
            f"The meta_agent model's context window ({context_window} tokens) is too small for context notes. "
            "Choose a model with a larger context window."
        )
    parent = min(parent, room)
    return ContextBudget(parent_tokens=parent, parent_overlap=min(overlap, parent // 6), batch_size=batch,
                         child_tokens=child_tokens)


class Contextualizer:

    def __init__(self, llm: ModelEndpoint, budget: Optional[ContextBudget] = None):
        self.llm = llm
        self.model = llm.model
        self.temperature = Config.LLM_TEMPERATURE
        self.max_concurrent = Config.LLM_MAX_CONCURRENT_CALLS
        self.max_chunks_per_batch = budget.batch_size if budget else Config.LLM_MAX_CHUNKS_PER_BATCH
        # Ollama serves only 2-4k tokens of context unless told otherwise, and silently cuts longer prompts
        self.extra_params = {}
        if budget and self.model.startswith(("ollama/", "ollama_chat/")):
            num_ctx = budget.call_tokens
            if llm.context_window:
                num_ctx = min(num_ctx, llm.context_window)
            self.extra_params["num_ctx"] = num_ctx

        self.semaphore = asyncio.Semaphore(self.max_concurrent)

        logger.info(
            f"Initialized contextualizer with model {self.model}, batch size {self.max_chunks_per_batch}, "
            f"max concurrent: {self.max_concurrent}{', ' + str(self.extra_params) if self.extra_params else ''}"
        )

    async def contextualize_hierarchical(
        self,
        context_chunks: List[ContextChunk],
        child_chunks: List[ChildChunk],
        book_metadata: dict
    ) -> List[ContextualizedChildChunk]:
        """
        Write a context note for every child, from its parent (each child belongs to exactly one parent).
        Raises if any batch fails after its retries, so no chunk is ever stored without its note.
        """
        logger.info(
            f"Starting hierarchical contextualization: "
            f"{len(context_chunks)} parents, {len(child_chunks)} children"
        )

        child_by_index = {child.index: child for child in child_chunks}

        tasks = []
        for context_chunk in context_chunks:
            children = [child_by_index[idx] for idx in context_chunk.child_indices]
            for i in range(0, len(children), self.max_chunks_per_batch):
                tasks.append(self._contextualize_batch_with_retry(
                    parent_text=context_chunk.text,
                    children=children[i:i + self.max_chunks_per_batch],
                    context_id=context_chunk.context_id,
                    book_metadata=book_metadata
                ))

        logger.info(f"Created {len(tasks)} batches for parallel processing")

        results = await asyncio.gather(*tasks, return_exceptions=True)

        failures = [r for r in results if isinstance(r, BaseException)]
        if failures:
            raise RuntimeError(
                f"Context notes failed for {len(failures)} of {len(results)} batches; "
                f"first error: {type(failures[0]).__name__}: {failures[0]}"
            ) from failures[0]

        all_contextualized = sorted((c for batch in results for c in batch), key=lambda c: c.index)
        if len(all_contextualized) != len(child_chunks):
            raise RuntimeError(
                f"Got context notes for {len(all_contextualized)} chunks, expected {len(child_chunks)}"
            )

        logger.info(f"Contextualized {len(all_contextualized)} chunks total")
        return all_contextualized

    async def _contextualize_batch_with_retry(
        self,
        parent_text: str,
        children: List[ChildChunk],
        context_id: int,
        book_metadata: dict,
        max_retries: int = 3
    ) -> List[ContextualizedChildChunk]:

        for attempt in range(max_retries):
            try:
                return await self._contextualize_batch(
                    parent_text=parent_text,
                    children=children,
                    context_id=context_id,
                    book_metadata=book_metadata
                )
            except Exception as e:
                logger.warning(
                    f"Contextualization attempt {attempt + 1}/{max_retries} failed (context {context_id}): {str(e)}"
                )
                # a bad key / unknown model won't fix itself; bad JSON, a wrong note count or a network hiccup might
                if is_permanent_api_error(e):
                    raise
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"All retry attempts failed for context {context_id}")
                    raise

    async def _contextualize_batch(
        self,
        parent_text: str,
        children: List[ChildChunk],
        context_id: int,
        book_metadata: dict
    ) -> List[ContextualizedChildChunk]:

        prompt = self._build_hierarchical_prompt(parent_text, children, book_metadata)

        # Call LLM with semaphore
        async with self.semaphore:
            logger.info(f"Sending {len(children)} chunks for contextualization (Context {context_id})")

            # strict JSON schema where the model supports it; otherwise the prompt asks for JSON
            # and parse_contextual_output validates it (a bad answer raises -> batch retry)
            extra = {"response_format": ContextualOutput} if self.llm.supports_json_schema else {}
            response = await litellm.acompletion(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                # room for every note: some providers default to a small output limit and cut the JSON
                max_tokens=len(children) * (Config.CONTEXT_DESCRIPTION_MAX_TOKENS + 50) + 200,
                **extra,
                **self.extra_params,
                **self.llm.call_kwargs(),
            )

        result = parse_contextual_output(response.choices[0].message.content)

        if len(result.contextual_chunks) != len(children):
            # notes could end up next to the wrong chunks: treat as a bad answer (retried)
            raise ValueError(
                f"Model returned {len(result.contextual_chunks)} descriptions for {len(children)} chunks"
            )

        contextualized = []
        for child, context_desc in zip(children, result.contextual_chunks):
            contextualized.append(ContextualizedChildChunk(
                index=child.index,
                parent_context_id=child.parent_context_id,
                original_text=child.original_text,
                context_description=context_desc,
                combined_text=f"{context_desc}\n\n{child.original_text}",
                start_index=child.start_index,
                end_index=child.end_index,
                token_count=child.token_count,
                metadata=book_metadata
            ))

        return contextualized

    def _build_hierarchical_prompt(
        self,
        parent_text: str,
        children: List[ChildChunk],
        book_metadata: dict
    ) -> str:
        """Title + WHOLE parent first (the same for every batch of this parent -> cacheable), then the chunks."""
        book_name = book_metadata.get('title', 'Unknown')
        author = book_metadata.get('author', '')

        book_info = f"Book: {book_name}"
        if author:
            book_info += f" by {author}"

        chunks_text = "\n\n".join([
            f"Chunk {idx + 1}:\n{child.original_text}"
            for idx, child in enumerate(children)
        ])

        return f"""{book_info}

<parent_context>
{parent_text}
</parent_context>

<child_chunks>
{chunks_text}
</child_chunks>"""
