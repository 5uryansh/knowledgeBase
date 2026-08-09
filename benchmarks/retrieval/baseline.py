"""The 'just use a bigger embedding model' baseline: pure top-k vector search.

Builds a second FAISS index over the *same* chunks as the hybrid index, using a
larger embedding model (bge-large) on the GPU, then serves plain vector search.
"""
from __future__ import annotations
import time

from sentence_transformers import SentenceTransformer

from src.retrieval.chunk_store import ChunkStore
from src.retrieval.vector_store import VectorStore
from . import config


class BaselineEmbedder:
    def __init__(self, model_name: str = config.BASELINE_EMBED_MODEL, device: str = "cuda") -> None:
        self._model = SentenceTransformer(model_name, device=device)

    def embed_texts(self, texts, batch_size: int = 128):
        return self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True,
            batch_size=batch_size, show_progress_bar=True,
        )

    def embed_query(self, query: str):
        return self._model.encode(query, normalize_embeddings=True, convert_to_numpy=True).tolist()


class BaselineRetriever:
    """Pure top-k dense retrieval with the big model. No graph."""

    def __init__(self, embedder: BaselineEmbedder, vector_store: VectorStore, chunk_store: ChunkStore) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._chunk_store = chunk_store

    def retrieve(self, query: str, top_k: int = config.TOP_K):
        query_embedding = self._embedder.embed_query(query)
        results = self._vector_store.search(query_embedding, k=top_k)
        ids = [chunk_id for chunk_id, _ in results]
        return self._chunk_store.get_many(ids)


def build_baseline_index(device: str = "cuda") -> None:
    """Embed every chunk with the big model and save a FAISS index next to the hybrid one."""
    config.BASELINE_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    chunk_store = ChunkStore.load(config.INDEX_DIR / "chunks.json")
    # Use the hybrid index's own id list so both indexes cover the exact same chunks.
    chunk_ids = (config.INDEX_DIR / "chunk_ids.txt").read_text(encoding="utf-8").splitlines()
    chunk_ids = [cid for cid in chunk_ids if cid]
    chunks = chunk_store.get_many(chunk_ids)
    texts = [chunk.text for chunk in chunks]
    ordered_ids = [chunk.id for chunk in chunks]

    print(f"Embedding {len(texts)} chunks with {config.BASELINE_EMBED_MODEL} on {device}...")
    embedder = BaselineEmbedder(device=device)
    t0 = time.time()
    embeddings = embedder.embed_texts(texts)
    print(f"Embedded in {time.time() - t0:.1f}s  (dim={embeddings.shape[1]})")

    store = VectorStore(dimension=int(embeddings.shape[1]))
    store.add(ordered_ids, embeddings.tolist())
    store.save(
        config.BASELINE_INDEX_DIR / "embeddings.faiss",
        config.BASELINE_INDEX_DIR / "chunk_ids.txt",
    )
    print(f"Saved baseline index to {config.BASELINE_INDEX_DIR}")


def load_baseline_retriever(device: str = "cuda") -> BaselineRetriever:
    embedder = BaselineEmbedder(device=device)
    vector_store = VectorStore.load(
        config.BASELINE_INDEX_DIR / "embeddings.faiss",
        config.BASELINE_INDEX_DIR / "chunk_ids.txt",
    )
    chunk_store = ChunkStore.load(config.INDEX_DIR / "chunks.json")
    return BaselineRetriever(embedder, vector_store, chunk_store)


if __name__ == "__main__":
    build_baseline_index()
