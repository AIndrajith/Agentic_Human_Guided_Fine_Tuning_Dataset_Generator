
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    PointStruct,
    SparseVector as QdrantSparseVector,
    Modifier
)
from workers.models import ContextualizedChunk, ContextualizedChildChunk, SparseVector
from workers.config import Config
import logging
import uuid

logger = logging.getLogger(__name__)


class StorageService:
    """One Qdrant collection per project; name and vector size come from web_api's processing config."""

    def __init__(self, collection_name: str, embedding_dimension: int):
        self.client = QdrantClient(url=Config.QDRANT_URL)
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension

        logger.info(f"Initialized storage service, Qdrant URL: {Config.QDRANT_URL}, collection: {collection_name}")

    async def ensure_collection_exists(self, collection_name: str):

        try:
            exists = self.client.collection_exists(collection_name)

            if exists:
                dense = self.client.get_collection(collection_name).config.params.vectors["dense"]
                if dense.size != self.embedding_dimension:
                    raise ValueError(
                        f"Collection {collection_name} holds {dense.size}-dim vectors, "
                        f"but the embedder makes {self.embedding_dimension}-dim vectors"
                    )
            else:
                logger.info(f"Creating Qdrant collection: {collection_name}")
                
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        "dense": VectorParams(
                            size=self.embedding_dimension,
                            distance=Distance.COSINE
                        )
                    },
                    sparse_vectors_config={
                        "sparse": SparseVectorParams(
                            index=SparseIndexParams(
                                on_disk=False  
                            ),
                            modifier=Modifier.IDF  
                        )
                    }
                )
                
                logger.info(f"Created collection {collection_name}")

        except Exception as e:
            logger.error(f"Failed to ensure collection exists: {str(e)}")
            raise
    
    async def store_chunks(
        self,
        chunks: List[ContextualizedChildChunk],
        dense_vectors: List[List[float]],
        sparse_vectors: List[SparseVector],
        document_id: str,
        project_id: str,
        book_metadata: dict = None,
        data_category: str = "fiction"
    ) -> List[str]:

        collection_name = self.collection_name

        await self.ensure_collection_exists(collection_name)

        if not (len(chunks) == len(dense_vectors) == len(sparse_vectors)):
            raise ValueError(
                f"Mismatched lengths: {len(chunks)} chunks, "
                f"{len(dense_vectors)} dense vectors, "
                f"{len(sparse_vectors)} sparse vectors"
            )
        
        points = []
        point_ids = []
        
        for idx, (chunk, dense_vec, sparse_vec) in enumerate(
            zip(chunks, dense_vectors, sparse_vectors)
        ):
            point_id = str(uuid.uuid4())
            point_ids.append(point_id)

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
            
            point = PointStruct(
                id=point_id,
                vector={
                    "dense": dense_vec,
                    "sparse": QdrantSparseVector(
                        indices=sparse_vec.indices,
                        values=sparse_vec.values
                    )
                },
                payload=payload
            )
            
            points.append(point)
        
        logger.info(f"Uploading {len(points)} points to Qdrant collection {collection_name}")
        
        self.client.upsert(
            collection_name=collection_name,
            points=points
        )
        
        logger.info(f"Successfully stored {len(points)} chunks in Qdrant")
        
        return point_ids
    
    async def delete_document_chunks(self, document_id: str):

        collection_name = self.collection_name
        if not self.client.collection_exists(collection_name):
            return

        self.client.delete(
            collection_name=collection_name,
            points_selector={
                "filter": {
                    "must": [
                        {
                            "key": "document_id",
                            "match": {
                                "value": document_id
                            }
                        }
                    ]
                }
            }
        )
        
        logger.info(f"Deleted chunks for document {document_id} from {collection_name}")
