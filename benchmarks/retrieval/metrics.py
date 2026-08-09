"""LLM-free structural metrics computed from two retrieved result sets.

These need no API and directly probe the connection thesis: whether the hybrid
surfaces different, more diverse, cross-conversation chunks than plain vector
search.
"""
from __future__ import annotations


def ids_of(chunks) -> list[str]:
    return [chunk.id for chunk in chunks]


def jaccard(a_ids, b_ids) -> float:
    a, b = set(a_ids), set(b_ids)
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def overlap_count(a_ids, b_ids) -> int:
    return len(set(a_ids) & set(b_ids))


def unique_to_first(a_ids, b_ids) -> int:
    """How many ids are in the first set but not the second."""
    return len(set(a_ids) - set(b_ids))


def source_diversity(chunks) -> int:
    """Number of distinct source documents represented in the result set."""
    return len({chunk.source.name for chunk in chunks})


def cross_source_rate(chunks) -> float:
    """Fraction of results from a different source document than the top hit."""
    if not chunks:
        return 0.0
    top_source = chunks[0].source.name
    return sum(1 for chunk in chunks if chunk.source.name != top_source) / len(chunks)


def graph_contribution(hybrid_ids, hybrid_semantic_ids) -> int:
    """How many of the hybrid's final results were NOT in its own semantic-only
    top-k — i.e. chunks the graph expansion + fusion actually added."""
    return len(set(hybrid_ids) - set(hybrid_semantic_ids))
