from chonkie import SentenceChunker
from typing import List, Tuple
from workers.models import Chunk, ContextChunk, ChildChunk
from workers.config import Config
from workers.services.chunk_assignment import assign_children_to_parents
from workers.utils.tokenizer import chunk_tokenizer
import logging

logger = logging.getLogger(__name__)


class ChunkingService:
    
    def __init__(
        self,
        parent_chunk_size=None,
        parent_overlap=None,
        child_chunk_size=None,
        child_overlap=None
    ):

        
        self.parent_chunk_size = parent_chunk_size or Config.PARENT_CHUNK_SIZE
        self.parent_overlap = parent_overlap or Config.PARENT_CHUNK_OVERLAP
        
        self.parent_chunker = SentenceChunker(
            tokenizer=chunk_tokenizer(),
            chunk_size=self.parent_chunk_size,
            chunk_overlap=self.parent_overlap,
            min_sentences_per_chunk=Config.CHONKIE_MIN_SENTENCES
        )
        
        self.child_chunk_size = child_chunk_size or Config.CHILD_CHUNK_SIZE
        self.child_overlap = child_overlap or Config.CHILD_CHUNK_OVERLAP
        
        self.child_chunker = SentenceChunker(
            tokenizer=chunk_tokenizer(),
            chunk_size=self.child_chunk_size,
            chunk_overlap=self.child_overlap,
            min_sentences_per_chunk=Config.CHONKIE_MIN_SENTENCES
        )
        
        logger.info(
            f"Initialized hierarchical chunking: "
            f"Parent={self.parent_chunk_size}/{self.parent_overlap}t, "
            f"Child={self.child_chunk_size}/{self.child_overlap}t"
        )
    
    def create_hierarchical_chunks(self, text: str) -> Tuple[List[ContextChunk], List[ChildChunk]]:
        """Parents (overlapping, for context) and children (cut once over the whole text, for retrieval).
        Every child belongs to exactly one parent; positions are counted from the start of the text."""

        logger.info(f"Creating hierarchical chunks for {len(text):,} characters")

        parent_chonkie_chunks = self.parent_chunker.chunk(text)
        logger.info(f"Created {len(parent_chonkie_chunks)} parent chunks")

        # children from the whole text, not per parent: text in a parent overlap is cut only once
        child_chonkie_chunks = self.child_chunker.chunk(text)
        logger.info(f"Created {len(child_chonkie_chunks)} child chunks")

        context_chunks = [
            ContextChunk(
                context_id=idx,
                text=parent_chunk.text,
                token_count=parent_chunk.token_count,
                start_index=parent_chunk.start_index,
                end_index=parent_chunk.end_index,
            )
            for idx, parent_chunk in enumerate(parent_chonkie_chunks)
        ]
        child_chunks = [
            ChildChunk(
                index=idx,
                parent_context_id=-1,   # set by assign_children_to_parents
                original_text=child_chunk.text,
                start_index=child_chunk.start_index,
                end_index=child_chunk.end_index,
                token_count=child_chunk.token_count,
            )
            for idx, child_chunk in enumerate(child_chonkie_chunks)
        ]
        assign_children_to_parents(context_chunks, child_chunks)

        logger.info(
            f"Hierarchical chunking complete: "
            f"{len(context_chunks)} parents, {len(child_chunks)} children"
        )
        
        return context_chunks, child_chunks
    
    def chunk_text(self, text: str) -> List[Chunk]:
        
        logger.warning("Using deprecated chunk_text() method. Use create_hierarchical_chunks() instead.")
        
        chonkie_chunks = self.child_chunker.chunk(text)
        
        chunks = [
            Chunk(
                index=idx,
                text=chunk.text,
                start_char=chunk.start_index,
                end_char=chunk.end_index,
                token_count=chunk.token_count,
                metadata=None
            )
            for idx, chunk in enumerate(chonkie_chunks)
        ]
        
        logger.info(f"Created {len(chunks)} chunks (legacy mode)")
        return chunks
    
    def get_chunker_metadata(self) -> dict:
        return {
            "service": "hierarchical_chonkie",
            "parent": {
                "chunk_size": self.parent_chunk_size,
                "overlap": self.parent_overlap,
                "tokenizer": Config.CHONKIE_TOKENIZER
            },
            "child": {
                "chunk_size": self.child_chunk_size,
                "overlap": self.child_overlap,
                "tokenizer": Config.CHONKIE_TOKENIZER
            }
        }

