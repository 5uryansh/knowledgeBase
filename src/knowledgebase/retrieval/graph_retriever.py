"""Graph retrieval with Personalized PageRank (HippoRAG-style).

The retrieval graph joins two layers:

    concept ── concept      (co-occurrence, NPMI-weighted ``strength``)
    concept ── chunk        (concept is linked in the chunk; IDF-weighted)
    chunk   ── chunk        (adjacent turns of the same conversation)

A query activates *seed* concepts (concepts named in the query, plus concepts in
the best first-stage hits) and seed chunks (the first-stage hits themselves).
Personalized PageRank then spreads that activation through the graph, so a chunk
can rank highly because it talks about something *connected* to the query —
even when it shares no words with it. That is what makes multi-hop questions
("what display did I use in the project where I also used a rotary encoder?")
answerable.

IDF weighting on concept→chunk edges and NPMI on concept→concept edges keep hub
concepts (your own name, "python", ...) from flooding everything.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from knowledgebase.graph.model import KnowledgeGraph
from knowledgebase.retrieval.chunk import Chunk
from knowledgebase.text import STOPWORDS, normalize_node_name, stem

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9+#.\-]*[A-Za-z0-9+#]|[A-Za-z0-9]")


@dataclass
class GraphRetrieverConfig:
    """Defaults were tuned on the LoCoMo *dev* split (benchmarks/tune.py) and then
    evaluated once on the held-out test split — see benchmarks/README.md."""

    damping: float = 0.7  # probability of continuing the walk at each step
    iterations: int = 30
    concept_concept: float = 0.5  # weight of co-occurrence (NPMI strength) edges
    concept_chunk: float = 1.0  # weight of concept-mentioned-in-chunk edges (x IDF)
    chunk_chunk: float = 1.0  # weight of adjacent-turn edges within a conversation
    max_concept_neighbors: int = 30
    query_seed_weight: float = 1.0  # concepts named in the query
    chunk_seed_weight: float = 0.25  # first-stage hits as seeds
    prf_concept_weight: float = 0.5  # concepts of first-stage hits (pseudo-relevance feedback)
    hub_df_ratio: float = 0.25  # concepts in more than this share of chunks get no seed mass
    # Semantic seeding: concepts whose label embedding is closest to the query. Off by
    # default — with static embeddings (wordllama) it hurt on the dev split; worth
    # enabling and re-tuning with a transformer embedder.
    semantic_seeds: int = 0
    semantic_seed_weight: float = 0.5
    semantic_min_sim: float = 0.35


def _surface_key(text: str) -> tuple[str, ...]:
    return tuple(stem(t) for t in _WORD.findall(text.lower()) if t not in STOPWORDS)


class ConceptMatcher:
    """Finds the concepts a free-text query refers to (exact/alias/stemmed n-gram match)."""

    def __init__(self, graph: KnowledgeGraph, max_ngram: int = 4):
        self.max_ngram = max_ngram
        self.index: dict[tuple[str, ...], set[str]] = defaultdict(set)
        for node_id, node in graph.nodes.items():
            names = {node.get("label", ""), node_id.replace("_", " "), *node.get("aliases", [])}
            for name in names:
                key = _surface_key(name)
                if key and len(key) <= max_ngram:
                    self.index[key].add(node_id)

    def match(self, query: str) -> dict[str, int]:
        """node id -> number of query tokens covered by the match (longer = more specific)."""
        words = [w for w in _WORD.findall(query.lower()) if w not in STOPWORDS]
        tokens = [stem(w) for w in words]
        found: dict[str, int] = {}
        for n in range(min(self.max_ngram, len(tokens)), 0, -1):
            for i in range(len(tokens) - n + 1):
                gram = tuple(tokens[i:i + n])
                if n == 1 and len(words[i]) < 3 and not words[i].isupper():
                    continue
                for node in self.index.get(gram, ()):
                    found[node] = max(found.get(node, 0), n)
        return found


class GraphRetriever:
    def __init__(self, graph: KnowledgeGraph, chunks: list[Chunk], config: GraphRetrieverConfig | None = None,
                 concept_vectors: np.ndarray | None = None):
        self.graph = graph
        self.chunks = chunks
        self.cfg = config or GraphRetrieverConfig()
        self.concepts = sorted(graph.nodes)
        self.concept_idx = {c: i for i, c in enumerate(self.concepts)}
        self.n_concepts = len(self.concepts)
        self.n_chunks = len(chunks)
        self.matcher = ConceptMatcher(graph)
        # Row i = embedding of concept label self.concepts[i] (optional; enables semantic seeding).
        self.concept_vectors = concept_vectors if concept_vectors is not None and len(concept_vectors) == self.n_concepts else None
        self._build()

    # ------------------------------------------------------------------ build
    def _build(self) -> None:
        nc, nk = self.n_concepts, self.n_chunks
        n = nc + nk
        rows, cols, vals = [], [], []

        def add(i: int, j: int, w: float) -> None:
            if w > 0 and i != j:
                rows.extend((i, j))
                cols.extend((j, i))
                vals.extend((w, w))

        # concept -> chunk, IDF weighted
        df = np.zeros(nc, dtype=np.float32)
        chunk_concepts: list[list[int]] = []
        for chunk in self.chunks:
            ids = [self.concept_idx[l] for l in chunk.links if l in self.concept_idx]
            chunk_concepts.append(ids)
            for i in ids:
                df[i] += 1
        self.df = df
        self.idf = np.log((nk + 1) / (df + 1)).astype(np.float32) + 0.1
        self.chunk_concepts = chunk_concepts
        for k, ids in enumerate(chunk_concepts):
            for i in ids:
                add(i, nc + k, self.cfg.concept_chunk * float(self.idf[i]))

        # concept -> concept, strongest neighbours only
        for node in self.concepts:
            i = self.concept_idx[node]
            for other, edge in self.graph.neighbors(node, limit=self.cfg.max_concept_neighbors):
                j = self.concept_idx.get(other)
                if j is not None and i < j:
                    add(i, j, self.cfg.concept_concept * float(edge.get("strength", 0)))

        # chunk -> next chunk in the same conversation
        for k in range(nk - 1):
            a, b = self.chunks[k], self.chunks[k + 1]
            if a.conversation_id == b.conversation_id:
                add(nc + k, nc + k + 1, self.cfg.chunk_chunk)

        A = sparse.csr_matrix((np.asarray(vals, dtype=np.float32), (rows, cols)), shape=(n, n))
        A.sum_duplicates()
        out_degree = np.asarray(A.sum(axis=1)).ravel()
        out_degree[out_degree == 0] = 1.0
        # Column-stochastic transition matrix for r <- (1-d) s + d * P r
        self.P = (sparse.diags(1.0 / out_degree) @ A).T.tocsr()
        self.hub = df > max(5.0, self.cfg.hub_df_ratio * nk)

    # ------------------------------------------------------------------ query
    def seeds(self, query: str, first_stage: list[tuple[int, float]] | None = None,
              query_vector: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, float]]:
        nc = self.n_concepts
        s = np.zeros(nc + self.n_chunks, dtype=np.float64)
        matched = self.matcher.match(query)
        query_concepts: dict[str, float] = {}
        for node, span in matched.items():
            i = self.concept_idx.get(node)
            if i is None or self.hub[i]:
                continue
            w = self.cfg.query_seed_weight * float(self.idf[i]) * span
            s[i] += w
            query_concepts[node] = w
        if query_vector is not None and self.concept_vectors is not None and self.cfg.semantic_seeds > 0:
            sims = self.concept_vectors @ query_vector.astype(np.float32)
            top = np.argpartition(-sims, min(self.cfg.semantic_seeds, nc) - 1)[: self.cfg.semantic_seeds]
            for i in top:
                sim = float(sims[i])
                if sim < self.cfg.semantic_min_sim or self.hub[i]:
                    continue
                w = self.cfg.semantic_seed_weight * sim * float(self.idf[i])
                s[i] += w
                node = self.concepts[i]
                query_concepts[node] = query_concepts.get(node, 0.0) + w
        if first_stage:
            for rank, (k, _score) in enumerate(first_stage):
                w = 1.0 / (rank + 1)
                s[nc + k] += self.cfg.chunk_seed_weight * w
                for i in self.chunk_concepts[k]:
                    if not self.hub[i]:
                        s[i] += self.cfg.prf_concept_weight * w * float(self.idf[i]) / max(len(self.chunk_concepts[k]), 1)
        return s, query_concepts

    def personalized_pagerank(self, seed: np.ndarray) -> np.ndarray:
        total = seed.sum()
        if total <= 0:
            return np.zeros_like(seed)
        s = seed / total
        r = s.copy()
        d = self.cfg.damping
        for _ in range(self.cfg.iterations):
            r_next = (1 - d) * s + d * (self.P @ r)
            if np.abs(r_next - r).sum() < 1e-7:
                r = r_next
                break
            r = r_next
        return r

    def score(self, query: str, first_stage: list[tuple[int, float]] | None = None,
              query_vector: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        """Returns (chunk scores, concept scores, query concepts)."""
        seed, query_concepts = self.seeds(query, first_stage, query_vector)
        r = self.personalized_pagerank(seed)
        return r[self.n_concepts:], r[: self.n_concepts], query_concepts

    def explain(self, chunk_index: int, concept_scores: np.ndarray, top: int = 3) -> list[str]:
        """Concepts that carried the most activation into this chunk — the 'why'."""
        ids = [i for i in self.chunk_concepts[chunk_index] if not self.hub[i]]
        ranked = sorted(ids, key=lambda i: -concept_scores[i] * self.idf[i])
        return [self.concepts[i] for i in ranked[:top] if concept_scores[i] > 0]


def idf_of(df: float, n: int) -> float:
    return math.log((n + 1) / (df + 1)) + 0.1


__all__ = ["GraphRetriever", "GraphRetrieverConfig", "ConceptMatcher", "normalize_node_name"]
