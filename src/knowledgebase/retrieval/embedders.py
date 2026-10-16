"""Dense embedding backends.

* ``sentence-transformers`` (default when installed): BAAI/bge-small-en-v1.5 —
  strong quality, needs torch + a one-off model download.
* ``wordllama``: 256-d static embeddings whose weights ship *inside* the pip wheel
  (~16 MB, no torch, no download, very fast on CPU). Lower quality, great for
  laptops / air-gapped machines / CI.
* ``none``: disable dense retrieval (BM25 + graph still work).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)

_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:  # pragma: no cover
        ...

    def embed_query(self, query: str) -> np.ndarray:  # pragma: no cover
        ...


def _normalize(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", batch_size: int = 64) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.name = f"st:{model_name}"
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self.batch_size = batch_size
        self._query_prefix = _BGE_QUERY_PREFIX if "bge" in model_name.lower() and "-en" in model_name.lower() else ""

    def embed(self, texts: list[str]) -> np.ndarray:
        return _normalize(self._model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                             convert_to_numpy=True, show_progress_bar=len(texts) > 500))

    def embed_query(self, query: str) -> np.ndarray:
        return _normalize(self._model.encode(self._query_prefix + query, normalize_embeddings=True,
                                             convert_to_numpy=True))


class WordLlamaEmbedder:
    def __init__(self) -> None:
        import wordllama
        from wordllama import WordLlama

        package_dir = Path(wordllama.__file__).parent
        try:  # weights + tokenizer ship inside the wheel; avoid any network access
            self._model = WordLlama.load(cache_dir=package_dir, disable_download=True)
        except FileNotFoundError:
            self._model = WordLlama.load()
        self.name = "wordllama:l2_supercat_256"
        self.dim = 256

    def embed(self, texts: list[str]) -> np.ndarray:
        out = [self._model.embed(texts[i:i + 512], norm=True) for i in range(0, len(texts), 512)]
        return _normalize(np.vstack(out)) if out else np.zeros((0, self.dim), dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return _normalize(self._model.embed([query], norm=True))[0]


def build_embedder(kind: str = "auto", model_name: str = "BAAI/bge-small-en-v1.5") -> Embedder | None:
    kind = (kind or "auto").lower()
    if kind == "none":
        return None
    if kind in ("auto", "sentence-transformers", "st"):
        try:
            return SentenceTransformerEmbedder(model_name)
        except Exception as exc:  # noqa: BLE001 - missing dependency or model download blocked
            if kind != "auto":
                raise
            log.warning("sentence-transformers unavailable (%s); trying wordllama.", exc.__class__.__name__)
    if kind in ("auto", "wordllama"):
        try:
            return WordLlamaEmbedder()
        except Exception as exc:  # noqa: BLE001
            if kind != "auto":
                raise
            log.warning("wordllama unavailable (%s); dense retrieval disabled.", exc.__class__.__name__)
    return None
