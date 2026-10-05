from chonkie import SentenceChunker, TableChunker, CodeChunker, MarkdownChef
from typing import List, Tuple, Dict
from workers.models import ContextChunk, ChildChunk
from workers.config import Config
from workers.services.chunk_assignment import assign_children_to_parents
from workers.utils.tokenizer import chunk_tokenizer
from dataclasses import dataclass
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class _MathChunk:
    """A math block is never split: it becomes one child as-is."""
    text: str
    token_count: int
    start_index: int = 0
    end_index: int = 0


class AcademicChunkingService:
    
    def __init__(
        self,
        parent_chunk_size=None,
        parent_overlap=None,
        child_chunk_size=None
    ):
        self.parent_chunk_size = parent_chunk_size or Config.ACADEMIC_PARENT_CHUNK_SIZE
        self.parent_overlap = parent_overlap or Config.ACADEMIC_PARENT_CHUNK_OVERLAP
        
        self.parent_chunker = SentenceChunker(
            tokenizer=chunk_tokenizer(),
            chunk_size=self.parent_chunk_size,
            chunk_overlap=self.parent_overlap,  
            min_sentences_per_chunk=Config.CHONKIE_MIN_SENTENCES
        )
        
        self.child_chunk_size = child_chunk_size or Config.ACADEMIC_CHILD_CHUNK_SIZE
        
        self.markdown_chef = MarkdownChef(tokenizer=chunk_tokenizer())
        
        logger.info(
            f"AcademicChunkingService initialized: "
            f"Parent={self.parent_chunk_size}/{self.parent_overlap}t, "
            f"Child={self.child_chunk_size}t (no overlap)"
        )
    
    def create_hierarchical_chunks(
        self,
        enriched_markdown: str
    ) -> Tuple[List[ContextChunk], List[ChildChunk]]:

        logger.info(f"Creating academic hierarchical chunks for {len(enriched_markdown):,} chars")
        
        doc = self.markdown_chef.parse(enriched_markdown)   # parse(text); process() expects a file path
        logger.info(
            f"MarkdownChef found: {len(doc.tables)} tables, "
            f"{len(doc.code)} code blocks"
        )

        math_blocks = self._extract_math_blocks(enriched_markdown)
        logger.info(f"Found {len(math_blocks)} math blocks")
        
        parent_chunks = self._chunk_parent_chunks(enriched_markdown)
        logger.info(f"Created {len(parent_chunks)} parent chunks")

        child_chunks = self._chunk_child_chunks_smart(
            enriched_markdown,
            parent_chunks,
            doc,
            math_blocks
        )
        logger.info(f"Created {len(child_chunks)} child chunks")
        
        return parent_chunks, child_chunks
    
    def _extract_math_blocks(self, markdown: str) -> List[Dict]:
        
        math_blocks = []
        

        block_pattern = r'\$\$(.+?)\$\$'
        for match in re.finditer(block_pattern, markdown, re.DOTALL):
            math_blocks.append({
                "content": match.group(0),
                "start": match.start(),
                "end": match.end(),
                "type": "block_math"
            })
        
        
        logger.debug(f"Extracted {len(math_blocks)} math blocks")
        return math_blocks
    
    def _chunk_parent_chunks(self, text: str) -> List[ContextChunk]:

        parent_chonkie_chunks = self.parent_chunker.chunk(text)
        
        context_chunks = []
        for idx, parent_chunk in enumerate(parent_chonkie_chunks):
            context_chunks.append(ContextChunk(
                context_id=idx,
                text=parent_chunk.text,
                token_count=parent_chunk.token_count,
                start_index=parent_chunk.start_index,
                end_index=parent_chunk.end_index,
                child_indices=[]  # Populated later
            ))
        
        return context_chunks
    
    def _chunk_child_chunks_smart(
        self,
        markdown: str,
        parent_chunks: List[ContextChunk],
        doc,  # MarkdownDocument from MarkdownChef
        math_blocks: List[Dict]
    ) -> List[ChildChunk]:
       
        all_child_chunks = []
        global_child_index = 0

        text_chunker = SentenceChunker(
            tokenizer=chunk_tokenizer(),
            chunk_size=self.child_chunk_size,
            chunk_overlap=0,  
            min_sentences_per_chunk=1
        )
        
        table_chunker = TableChunker(
            tokenizer="row",
            chunk_size=Config.ACADEMIC_TABLE_CHUNK_MAX_ROWS
        )

        # segment the WHOLE document once (tables/code/math/text), then give each child one parent:
        # chunking per parent range cut every overlap zone twice
        segments = self._build_segments_for_range(markdown, 0, len(markdown), doc, math_blocks)

        for segment in segments:
            seg_type = segment["type"]
            content = segment["content"]

            if not content or not content.strip():
                continue

            chunks = []

            if seg_type == "text":
                chunks = text_chunker.chunk(content)

            elif seg_type == "table":
                chunks = table_chunker.chunk(content)

            elif seg_type == "code":
                code_chunker = CodeChunker(
                    language=segment.get("language", "python"),
                    tokenizer=chunk_tokenizer(),
                    chunk_size=Config.ACADEMIC_CODE_CHUNK_SIZE,
                    include_nodes=False
                )
                chunks = code_chunker.chunk(content)

            elif seg_type == "block_math":
                chunks = [_MathChunk(
                    text=content,
                    token_count=len(chunk_tokenizer().encode(content)),
                    start_index=0,
                    end_index=len(content)
                )]

            for chunk in chunks:
                all_child_chunks.append(ChildChunk(
                    index=global_child_index,
                    parent_context_id=-1,   # set by assign_children_to_parents
                    original_text=chunk.text,
                    start_index=segment["start"] + chunk.start_index,
                    end_index=segment["start"] + chunk.end_index,
                    token_count=chunk.token_count
                ))
                global_child_index += 1

        assign_children_to_parents(parent_chunks, all_child_chunks)
        return all_child_chunks
    
    def _build_segments_for_range(
        self,
        markdown: str,
        range_start: int,
        range_end: int,
        doc,
        math_blocks: List[Dict]
    ) -> List[Dict]:

        components = []

        for table in doc.tables:
            if self._overlaps_range(table.start_index, table.end_index, range_start, range_end):
                components.append({
                    "type": "table",
                    "start": max(table.start_index, range_start),
                    "end": min(table.end_index, range_end),
                    "content": table.content,
                    "obj": table
                })

        for code in doc.code:
            if self._overlaps_range(code.start_index, code.end_index, range_start, range_end):
                components.append({
                    "type": "code",
                    "start": max(code.start_index, range_start),
                    "end": min(code.end_index, range_end),
                    "content": code.content,
                    "language": code.language,
                    "obj": code
                })

        for math in math_blocks:
            if self._overlaps_range(math["start"], math["end"], range_start, range_end):
                components.append({
                    "type": "block_math",
                    "start": max(math["start"], range_start),
                    "end": min(math["end"], range_end),
                    "content": math["content"]
                })

        boundaries = [range_start, range_end]
        for comp in components:
            boundaries.extend([comp["start"], comp["end"]])
        
        boundaries = sorted(set(boundaries))

        component_map = {comp["start"]: comp for comp in components}

        segments = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            
            if start in component_map:
                comp = component_map[start]
                segments.append({
                    "type": comp["type"],
                    "start": start,
                    "end": end,
                    "content": comp["content"],
                    "language": comp.get("language")
                })
            else:
                segments.append({
                    "type": "text",
                    "start": start,
                    "end": end,
                    "content": markdown[start:end]
                })
        
        return segments
    
    def _overlaps_range(
        self,
        comp_start: int,
        comp_end: int,
        range_start: int,
        range_end: int
    ) -> bool:
        return not (comp_end <= range_start or comp_start >= range_end)
    
    def get_chunker_metadata(self) -> dict:

        return {
            "service": "academic_chonkie_pre_segmentation",
            "parent": {
                "chunk_size": self.parent_chunk_size,
                "overlap": self.parent_overlap,
                "tokenizer": Config.CHONKIE_TOKENIZER
            },
            "child": {
                "chunk_size": self.child_chunk_size,
                "overlap": 0, 
                "tokenizer": Config.CHONKIE_TOKENIZER
            },
            "components": {
                "tables": f"TableChunker(max_rows={Config.ACADEMIC_TABLE_CHUNK_MAX_ROWS})",
                "code": f"CodeChunker(size={Config.ACADEMIC_CODE_CHUNK_SIZE})",
                "math": "Keep whole (never split)"
            }
        }
