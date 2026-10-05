"""
Service for generating embeddings (any provider, via LiteLLM).
"""

from typing import List
import litellm
from workers.config import Config
from workers.models import EmbedderEndpoint
from workers.utils.retry import retry_async
import logging

logger = logging.getLogger(__name__)


def _vector(item) -> List[float]:
    return item["embedding"] if isinstance(item, dict) else item.embedding


class EmbeddingService:

    def __init__(self, embedder: EmbedderEndpoint):
        self.embedder = embedder
        self.model = embedder.model
        self.dimension = embedder.dimension   # measured by web_api when the embedder was saved
        self.batch_size = Config.EMBEDDING_BATCH_SIZE

        logger.info(f"Initialized embedding service with model {self.model} ({self.dimension}d)")

    async def _embed(self, texts: List[str]) -> List[List[float]]:
        response = await retry_async(
            litellm.aembedding, model=self.model, input=texts, what="embedding batch", **self.embedder.call_kwargs()
        )
        # keep input order even if the provider returns items out of order
        items = sorted(response.data, key=lambda i: i["index"] if isinstance(i, dict) else i.index)
        vectors = [_vector(item) for item in items]
        if vectors and len(vectors[0]) != self.dimension:
            raise ValueError(
                f"{self.model} returned {len(vectors[0])}-dim vectors, but the project expects {self.dimension}. "
                "Save the embedder step again."
            )
        return vectors

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            logger.info(f"Generating embeddings for batch {i//self.batch_size + 1} ({len(batch)} texts)")
            all_embeddings.extend(await self._embed(batch))

        logger.info(f"Generated {len(all_embeddings)} embeddings")
        return all_embeddings

    async def generate_single_embedding(self, text: str) -> List[float]:
        return (await self._embed([text]))[0]
