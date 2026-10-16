from __future__ import annotations
from pathlib import Path
import faiss
import numpy as np


class VectorStore:
    def __init__(self, dimension: int) -> None:
        self._index = faiss.IndexFlatIP(dimension)
        self._chunk_ids: list[str] = []

    def add(self, chunk_ids: list[str], embeddings: list[list[float]]) -> None:
        vectors = np.asarray(embeddings, dtype=np.float32)
        self._index.add(vectors)
        self._chunk_ids.extend(chunk_ids)

    def search(self, query_embedding: list[float], k: int = 10) -> list[tuple[str, float]]:
        if self._index.ntotal == 0:
            return []

        query = np.asarray([query_embedding], dtype=np.float32)
        scores, indices = self._index.search(query, k)
        results: list[tuple[str, float]] = []

        for score, index in zip(scores[0], indices[0]):
            if index == -1:
                continue
            results.append((self._chunk_ids[index], float(score)))

        return results

    def save(self, index_path: Path, metadata_path: Path) -> None:
        faiss.write_index(self._index, str(index_path))
        metadata_path.write_text("\n".join(self._chunk_ids), encoding="utf-8")

    @classmethod
    def load(cls, index_path: Path, metadata_path: Path) -> "VectorStore":
        index = faiss.read_index(str(index_path))
        store = cls(index.d)
        store._index = index
        store._chunk_ids = metadata_path.read_text(encoding="utf-8").splitlines()
        return store