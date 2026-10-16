from __future__ import annotations
from sentence_transformers import SentenceTransformer
from src.retrieval.chunk import Chunk


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5",) -> None:
        self._model = SentenceTransformer(model_name)

    def embed(self, chunks: list[Chunk]) -> list[list[float]]:
        """
        Generate embeddings for a list of chunks.
        """
        texts = [chunk.text for chunk in chunks]
        embeddings = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """
        Generate an embedding for a search query.
        """
        embedding = self._model.encode(query, normalize_embeddings=True, convert_to_numpy=True)
        return embedding.tolist()