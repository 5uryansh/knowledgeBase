"""Exact cosine search over normalised embeddings (numpy; FAISS used when installed and large)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


class VectorStore:
    def __init__(self, matrix: np.ndarray | None = None, dim: int = 0) -> None:
        self.matrix = matrix if matrix is not None else np.zeros((0, dim), dtype=np.float32)
        self._faiss = None
        if len(self.matrix) > 200_000:
            try:
                import faiss

                self._faiss = faiss.IndexFlatIP(self.matrix.shape[1])
                self._faiss.add(self.matrix)
            except ImportError:
                pass

    def __len__(self) -> int:
        return len(self.matrix)

    def scores(self, query: np.ndarray) -> np.ndarray:
        """Cosine similarity of the query with every row."""
        if not len(self.matrix):
            return np.zeros(0, dtype=np.float32)
        return self.matrix @ query.astype(np.float32)

    def search(self, query: np.ndarray, k: int = 10) -> list[tuple[int, float]]:
        if not len(self.matrix):
            return []
        k = min(k, len(self.matrix))
        if self._faiss is not None:
            scores, idx = self._faiss.search(query.reshape(1, -1).astype(np.float32), k)
            return [(int(i), float(s)) for i, s in zip(idx[0], scores[0]) if i >= 0]
        s = self.scores(query)
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.argsort(-s[top])]
        return [(int(i), float(s[i])) for i in top]

    def save(self, path: Path) -> None:
        np.save(path, self.matrix)

    @classmethod
    def load(cls, path: Path) -> "VectorStore":
        return cls(np.load(path))
