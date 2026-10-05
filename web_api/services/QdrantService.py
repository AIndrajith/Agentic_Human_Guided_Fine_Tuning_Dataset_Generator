"""Qdrant cleanup. The worker creates and fills each project's collection; web_api only deletes."""
import logging
import uuid

from qdrant_client import AsyncQdrantClient, models

from web_api.core.config import get_settings

logger = logging.getLogger(__name__)


def collection_name(project_id: uuid.UUID) -> str:
    """One collection per project, so each project can use its own embedder and vector size."""
    return f"project_{project_id.hex}"


class QdrantService:
    def __init__(self, url: str):
        self.url = url
        self._client: AsyncQdrantClient | None = None

    @property
    def client(self) -> AsyncQdrantClient:
        if self._client is None:
            self._client = AsyncQdrantClient(url=self.url, timeout=10)
        return self._client

    async def drop_project(self, project_id: uuid.UUID) -> None:
        """Delete the project's collection. Logs instead of raising: the project row is already gone."""
        name = collection_name(project_id)
        try:
            if await self.client.collection_exists(name):
                await self.client.delete_collection(name)
                logger.info("Deleted Qdrant collection %s", name)
        except Exception:
            logger.exception("Project deleted but Qdrant collection remains: %s", name)

    async def drop_document(self, project_id: uuid.UUID, document_id: uuid.UUID) -> None:
        """Delete one document's vectors from its project's collection. Logs instead of raising."""
        name = collection_name(project_id)
        try:
            if await self.client.collection_exists(name):
                await self.client.delete(
                    collection_name=name,
                    points_selector=models.FilterSelector(filter=models.Filter(must=[
                        models.FieldCondition(key="document_id", match=models.MatchValue(value=str(document_id))),
                    ])),
                )
        except Exception:
            logger.exception("Document deleted but its Qdrant vectors remain: %s in %s", document_id, name)


qdrant_service = QdrantService(get_settings().QDRANT_URL)
