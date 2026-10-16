"""Faithful re-implementation of the original (pre-0.2) retriever, for comparison.

Original behaviour (``src/retrieval/retriever.py`` before this change):

1. dense top-10
2. collect every entity linked in those chunks
3. expand each entity with the *first 5* neighbours of the unweighted adjacency list
4. add *every* chunk that mentions any expanded entity
5. return dense hits followed by the expansion, unranked

Because dense hits come first, its top-10 is identical to dense top-10; the graph
step only appends a large, unordered tail. The benchmark reports how big that
tail is and how much of the evidence it actually contains.
"""
from __future__ import annotations

from collections import defaultdict


class LegacyRetriever:
    def __init__(self, index):
        self.index = index
        adjacency: dict[str, list[str]] = defaultdict(list)
        for e in index.graph.edges:  # insertion order, like graph_edges.json
            adjacency[e["source"]].append(e["target"])
            adjacency[e["target"]].append(e["source"])
        self.adjacency = adjacency

    def search(self, query: str, top_k: int = 10) -> list[int]:
        idx = self.index
        if idx.embedder is None:
            return []
        hits = [i for i, _ in idx.vectors.search(idx.embedder.embed_query(query), k=top_k)]
        entities = {link for i in hits for link in idx.chunks[i].links}
        expanded = set(entities)
        for e in list(entities):
            expanded.update(self.adjacency.get(e, [])[:5])
        related = sorted({i for e in expanded for i in idx.by_concept.get(e, [])})
        seen, merged = set(), []
        for i in hits + related:
            if i not in seen:
                seen.add(i)
                merged.append(i)
        return merged
