"""
Service for generating embeddings.
"""

from typing import List
from openai import AsyncOpenAI
from workers.config import Config
import logging

logger = logging.getLogger(__name__)


class EmbeddingService:

    def __init__(self, api_key: str = None, model: str = None, base_url: str = None):
        resolved_key = api_key or Config.OPENAI_API_KEY
        self.client = AsyncOpenAI(
            api_key=resolved_key,
            **({"base_url": base_url} if base_url else {})
        )
        self.model = model or Config.EMBEDDING_MODEL
        self.dimension = Config.EMBEDDING_DIMENSION
        self.batch_size = Config.EMBEDDING_BATCH_SIZE

        logger.info(f"Initialized embedding service with model {self.model}")
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            logger.info(f"Generating embeddings for batch {i//self.batch_size + 1} ({len(batch)} texts)")
            
            response = await self.client.embeddings.create(
                model=self.model,
                input=batch
            )
            
            # Extract embeddings in order
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        
        logger.info(f"Generated {len(all_embeddings)} embeddings")
        return all_embeddings
    
    async def generate_single_embedding(self, text: str) -> List[float]:

        response = await self.client.embeddings.create(
            model=self.model,
            input=[text]
        )
        
        return response.data[0].embedding
