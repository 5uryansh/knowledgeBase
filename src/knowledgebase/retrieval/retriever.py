from __future__ import annotations
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

    def retrieve(self, query: str, top_k: int = 10) -> list[Chunk]:
        # Step 1: Semantic search
        query_embedding = self._embedder.embed_query(query)
        search_results = self._vector_store.search(query_embedding, k=top_k)
        chunk_ids = [chunk_id for chunk_id, _ in search_results]
        chunks = self._chunk_store.get_many(chunk_ids)

        # Step 2: Collect entities
        entities = set()
        for chunk in chunks:
            entities.update(chunk.links)

        # Step 3: Expand through graph
        expanded_entities = self._graph_expander.expand(list(entities))

        # Step 4: Retrieve related chunks
        related_chunk_ids = self._entity_index.lookup(expanded_entities)

        related_chunks = self._chunk_store.get_many(related_chunk_ids)

        # Step 5: Merge & deduplicate
        merged: dict[str, Chunk] = {}
        for chunk in chunks + related_chunks:
            merged[chunk.id] = chunk

        return list(merged.values())