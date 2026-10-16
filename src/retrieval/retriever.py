from __future__ import annotations
from collections import defaultdict
from src.retrieval.chunk import Chunk
from src.retrieval.chunk_store import ChunkStore
from src.retrieval.embedder import Embedder
from src.retrieval.entity_index import EntityIndex
from src.retrieval.graph_expander import GraphExpander
from src.retrieval.vector_store import VectorStore


class Retriever:
    def __init__(self, embedder: Embedder, vector_store: VectorStore, chunk_store: ChunkStore, entity_index: EntityIndex, graph_expander: GraphExpander) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._chunk_store = chunk_store
        self._entity_index = entity_index
        self._graph_expander = graph_expander

    @staticmethod
    def _reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = 60) -> list[str]:
        """Fuse several ranked id lists into one, by reciprocal rank.

        Each list contributes 1/(k + rank) to an id's score, so an id ranked
        highly in either list rises without needing score-scale normalization
        between the (very different) semantic-similarity and graph-connection
        signals.
        """
        scores: dict[str, float] = defaultdict(float)
        for ranked in ranked_lists:
            for rank, item_id in enumerate(ranked, start=1):
                scores[item_id] += 1.0 / (k + rank)
        return sorted(scores, key=scores.get, reverse=True)

    def retrieve(self, query: str, top_k: int = 10, candidate_k: int = 50) -> list[Chunk]:
        # Step 1: Semantic search — a candidate pool larger than top_k so the
        # fusion has material to re-rank.
        query_embedding = self._embedder.embed_query(query)
        semantic_results = self._vector_store.search(query_embedding, k=candidate_k)
        semantic_ids = [chunk_id for chunk_id, _ in semantic_results]

        # Step 2: Seed graph expansion from the entities in the strongest
        # semantic matches only (weak matches would pollute the expansion).
        seed_chunks = self._chunk_store.get_many(semantic_ids[:top_k])
        seed_entities: set[str] = set()
        for chunk in seed_chunks:
            seed_entities.update(chunk.links)

        # Step 3: Expand through the co-occurrence graph.
        expanded_entities = self._graph_expander.expand(list(seed_entities))

        # Step 4: Graph-connected chunks, ranked by connection strength
        # (how many query-relevant entities each one shares).
        related_ids = self._entity_index.lookup(expanded_entities)
        related_chunks = self._chunk_store.get_many(related_ids)
        related_chunks.sort(
            key=lambda chunk: len(set(chunk.links) & expanded_entities),
            reverse=True,
        )
        graph_ids = [chunk.id for chunk in related_chunks]

        # Step 5: Fuse the semantic and graph rankings, then take top_k.
        fused_ids = self._reciprocal_rank_fusion([semantic_ids, graph_ids])
        return self._chunk_store.get_many(fused_ids[:top_k])
