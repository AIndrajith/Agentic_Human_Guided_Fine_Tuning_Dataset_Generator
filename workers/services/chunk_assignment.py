"""Give every child chunk exactly one parent.

Children are cut once over the whole document (no duplicates, positions counted from the document start).
Parents overlap, so a child can sit inside two parents; it goes to the one where it is closest to the
middle, which gives the context-note model the most text on both sides of it.
"""
from typing import List

from workers.models import ChildChunk, ContextChunk


def assign_children_to_parents(parents: List[ContextChunk], children: List[ChildChunk]) -> None:
    """Sets child.parent_context_id and fills parent.child_indices (in document order)."""
    if not parents:
        raise ValueError("No parent chunks to assign children to")
    for parent in parents:
        parent.child_indices = []

    for child in children:
        middle = (child.start_index + child.end_index) / 2
        containing = [p for p in parents if p.start_index <= child.start_index and child.end_index <= p.end_index]
        if containing:
            best = min(containing, key=lambda p: abs((p.start_index + p.end_index) / 2 - middle))
        else:
            # rare (a very long sentence across a parent edge): take the parent covering most of it
            best = max(parents, key=lambda p: min(p.end_index, child.end_index) - max(p.start_index, child.start_index))
        child.parent_context_id = best.context_id
        best.child_indices.append(child.index)
