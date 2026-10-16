"""Hybrid retrieval: BM25 + dense + graph (Personalized PageRank), fused with weighted RRF.

Modes (all share one index, which keeps benchmark ablations honest):

* ``bm25``          lexical only
* ``dense``         embeddings only
* ``hybrid``        BM25 + dense
* ``graph``         PPR seeded only by concepts named in the query
* ``graph_hybrid``  BM25 + dense, then PPR seeded by the query *and* the first-stage
                    hits, all three fused (default)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from knowledgebase.retrieval.chunk import Chunk

MODES = ("bm25", "dense", "hybrid", "graph", "graph_hybrid")


@dataclass
class FusionConfig:
    rrf_k: int = 60
    w_bm25: float = 1.0
    w_dense: float = 1.0
    w_graph: float = 1.0
    first_stage_k: int = 20
    candidates: int = 200


@dataclass
class SearchResult:
    chunk: Chunk
    score: float
    rank: int
    signals: dict[str, int] = field(default_factory=dict)  # signal -> rank (1-based) in that list
    via: list[str] = field(default_factory=list)  # concepts that connected the query to this chunk

    def to_dict(self, max_chars: int | None = None) -> dict:
        text = self.chunk.plain
        if max_chars and len(text) > max_chars:
            text = text[:max_chars].rsplit(" ", 1)[0] + " …"
        return {
            "rank": self.rank,
            "score": round(self.score, 6),
            "chunk_id": self.chunk.id,
            "conversation_id": self.chunk.conversation_id,
            "title": self.chunk.title,
            "platform": self.chunk.platform,
            "date": self.chunk.date,
            "source": self.chunk.source,
            "text": text,
            "concepts": self.chunk.links,
            "signals": self.signals,
            "via": self.via,
        }


def _ranking(scores: np.ndarray, limit: int) -> list[int]:
    if scores is None or not len(scores):
        return []
    limit = min(limit, len(scores))
    idx = np.argpartition(-scores, limit - 1)[:limit]
    idx = idx[np.argsort(-scores[idx])]
    return [int(i) for i in idx if scores[i] > 0]


class HybridRetriever:
    def __init__(self, index, fusion: FusionConfig | None = None):
        self.index = index
        self.fusion = fusion or FusionConfig()

    # ------------------------------------------------------------------ signals
    def _bm25(self, query: str) -> np.ndarray:
        return self.index.bm25.scores(query)

    def _dense(self, query: str) -> np.ndarray | None:
        if self.index.embedder is None or self.index.vectors is None or not len(self.index.vectors):
            return None
        return self.index.vectors.scores(self.index.embedder.embed_query(query))

    def _rrf(self, rankings: list[tuple[list[int], float]]) -> dict[int, float]:
        fused: dict[int, float] = {}
        for ranking, weight in rankings:
            for r, i in enumerate(ranking):
                fused[i] = fused.get(i, 0.0) + weight / (self.fusion.rrf_k + r + 1)
        return fused

    # ------------------------------------------------------------------ search
    def search(self, query: str, k: int = 10, mode: str = "graph_hybrid") -> list[SearchResult]:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")
        f = self.fusion
        n = len(self.index.chunks)
        if n == 0:
            return []
        signals: dict[str, list[int]] = {}
        concept_scores = None

        if mode in ("bm25", "hybrid", "graph_hybrid"):
            signals["bm25"] = _ranking(self._bm25(query), f.candidates)
        if mode in ("dense", "hybrid", "graph_hybrid"):
            dense = self._dense(query)
            if dense is not None:
                signals["dense"] = _ranking(dense, f.candidates)

        if mode in ("graph", "graph_hybrid"):
            first_stage = None
            if mode == "graph_hybrid":
                fs = self._rrf([(signals.get("bm25", []), f.w_bm25), (signals.get("dense", []), f.w_dense)])
                top = sorted(fs.items(), key=lambda t: -t[1])[: f.first_stage_k]
                first_stage = [(i, s) for i, s in top]
            chunk_scores, concept_scores, _ = self.index.graph_retriever.score(query, first_stage)
            signals["graph"] = _ranking(chunk_scores, f.candidates)

        weights = {"bm25": f.w_bm25, "dense": f.w_dense, "graph": f.w_graph}
        if len(signals) == 1:
            (name, ranking), = signals.items()
            fused = {i: 1.0 / (f.rrf_k + r + 1) for r, i in enumerate(ranking)}
        else:
            fused = self._rrf([(ranking, weights[name]) for name, ranking in signals.items()])

        rank_of = {name: {i: r + 1 for r, i in enumerate(ranking)} for name, ranking in signals.items()}
        top = sorted(fused.items(), key=lambda t: -t[1])[:k]
        results = []
        for r, (i, score) in enumerate(top, start=1):
            via = self.index.graph_retriever.explain(i, concept_scores) if concept_scores is not None else []
            results.append(SearchResult(
                chunk=self.index.chunks[i],
                score=score,
                rank=r,
                signals={name: ranks[i] for name, ranks in rank_of.items() if i in ranks},
                via=via,
            ))
        return results

    def timed_search(self, query: str, k: int = 10, mode: str = "graph_hybrid") -> tuple[list[SearchResult], float]:
        t0 = time.perf_counter()
        results = self.search(query, k=k, mode=mode)
        return results, (time.perf_counter() - t0) * 1000.0
