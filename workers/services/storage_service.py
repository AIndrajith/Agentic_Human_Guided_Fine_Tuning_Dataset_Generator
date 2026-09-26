
import asyncio
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Document,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    Modifier,
    PayloadSchemaType,
    PointStruct,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)
from workers.models import ContextualizedChildChunk
from workers.config import Config
from workers.utils.retry import retry_async
import logging
import uuid

logger = logging.getLogger(__name__)

# fixed namespace: the same document + chunk number always gives the same point id
_POINT_NAMESPACE = uuid.UUID("6f1c2b8e-3d4a-5e6f-8a9b-0c1d2e3f4a5b")

MIN_QDRANT_VERSION = (1, 15, 2)   # built-in BM25 ("qdrant/bm25")


def _version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("-")[0].split(".")[:3])


def point_id(document_id: str, chunk_index: int) -> str:
    """Deterministic id, so reprocessing a document overwrites its points instead of duplicating them."""
    return str(uuid.uuid5(_POINT_NAMESPACE, f"{document_id}:{chunk_index}"))


class StorageService:
    """One Qdrant collection per project; name and vector size come from web_api's processing config.
    The Qdrant client is sync, so every call runs in a thread (doesn't block the event loop) with retries."""

    def __init__(self, collection_name: str, embedding_dimension: int):
        # cloud_inference=True: send BM25 Documents to the server as text (Qdrant computes the vectors)
        # instead of trying to embed them locally with FastEmbed. Works on self-hosted Qdrant too.
        self.client = QdrantClient(url=Config.QDRANT_URL, cloud_inference=True)
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        self.batch_size = Config.QDRANT_UPSERT_BATCH_SIZE

        logger.info(f"Initialized storage service, Qdrant URL: {Config.QDRANT_URL}, collection: {collection_name}")

    async def _call(self, what: str, fn, *args, **kwargs):
        return await retry_async(asyncio.to_thread, fn, *args, what=f"Qdrant {what}", **kwargs)

    async def ensure_collection_exists(self):
        name = self.collection_name
        if await self._call("collection_exists", self.client.collection_exists, name):
            info = await self._call("get_collection", self.client.get_collection, name)
            dense = info.config.params.vectors["dense"]
            if dense.size != self.embedding_dimension:
                raise ValueError(
                    f"Collection {name} holds {dense.size}-dim vectors, "
                    f"but the embedder makes {self.embedding_dimension}-dim vectors"
                )
            return

        server = (await self._call("info", self.client.info)).version
        if _version_tuple(server) < MIN_QDRANT_VERSION:
            raise RuntimeError(
                f"Qdrant {server} is too old: BM25 needs 1.15.2 or newer. "
                "Update the image (docker compose pull qdrant) and recreate the container."
            )

        logger.info(f"Creating Qdrant collection: {name}")
        await self._call(
            "create_collection",
            self.client.create_collection,
            collection_name=name,
            vectors_config={"dense": VectorParams(size=self.embedding_dimension, distance=Distance.COSINE)},
            sparse_vectors_config={
                "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False), modifier=Modifier.IDF)
            },
        )
        # fast "delete / filter by document"
        await self._call(
            "create_payload_index",
            self.client.create_payload_index,
            collection_name=name, field_name="document_id", field_schema=PayloadSchemaType.KEYWORD,
        )
        logger.info(f"Created collection {name}")

    async def store_chunks(
        self,
        chunks: List[ContextualizedChildChunk],
        dense_vectors: List[List[float]],
        sparse_vectors: List[Document],   # BM25 documents from BM25Service; Qdrant computes the vectors
        document_id: str,
        project_id: str,
        book_metadata: dict = None,
        data_category: str = "fiction"
    ) -> List[str]:
        """Replace the document's points: delete the old ones, then upsert in batches."""
        if not (len(chunks) == len(dense_vectors) == len(sparse_vectors)):
            raise ValueError(
                f"Mismatched lengths: {len(chunks)} chunks, "
                f"{len(dense_vectors)} dense vectors, "
                f"{len(sparse_vectors)} sparse vectors"
            )

        await self.ensure_collection_exists()
        # a re-run may produce fewer chunks than last time; drop everything the document had
        await self.delete_document_chunks(document_id)

        points = []
        for chunk, dense_vec, sparse_vec in zip(chunks, dense_vectors, sparse_vectors):
            payload = {
                "original_text": chunk.original_text,
                "context_description": chunk.context_description,
                "combined_text": chunk.combined_text,
                "document_id": document_id,
                "project_id": project_id,
                "data_category": data_category,
                "chunk_index": chunk.index,
                "parent_context_id": chunk.parent_context_id,
                "start_index": chunk.start_index,
                "end_index": chunk.end_index,
                "token_count": chunk.token_count,
                "metadata": chunk.metadata or {}
            }
            if book_metadata:
                payload["book_metadata"] = book_metadata

            points.append(PointStruct(
                id=point_id(document_id, chunk.index),
                vector={"dense": dense_vec, "sparse": sparse_vec},
                payload=payload,
            ))

        logger.info(f"Uploading {len(points)} points to {self.collection_name} in batches of {self.batch_size}")
        for start in range(0, len(points), self.batch_size):
            batch = points[start:start + self.batch_size]
            await self._call(
                f"upsert {start}-{start + len(batch)}",
                self.client.upsert, collection_name=self.collection_name, points=batch, wait=True,
            )

        logger.info(f"Successfully stored {len(points)} chunks in Qdrant")
        return [str(p.id) for p in points]

    async def delete_document_chunks(self, document_id: str):
        name = self.collection_name
        if not await self._call("collection_exists", self.client.collection_exists, name):
            return
        await self._call(
            "delete",
            self.client.delete,
            collection_name=name,
            points_selector=FilterSelector(filter=Filter(must=[
                FieldCondition(key="document_id", match=MatchValue(value=document_id)),
            ])),
            wait=True,
        )
        logger.info(f"Deleted old chunks for document {document_id} from {name}")
