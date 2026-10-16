"""The on-disk index: chunks + embeddings + graph, loaded as one object.

Files in ``<vault>/indexes``::

    manifest.json        build metadata (embedder, counts, versions)
    chunks.jsonl         every chunk (text with [[links]] + metadata)
    embeddings.npy       float32, row i = chunk i (absent if dense is disabled)
    graph.json           nodes / edges / relations / communities / discoveries
    graph_edges.json     legacy adjacency export (backwards compatible)
    graph_relations.json legacy typed-relations export
    entity_index.json    concept -> chunk ids
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from knowledgebase.graph.model import KnowledgeGraph
from knowledgebase.retrieval.bm25 import BM25Index
from knowledgebase.retrieval.chunk import Chunk
from knowledgebase.retrieval.embedders import Embedder, build_embedder
from knowledgebase.retrieval.graph_retriever import GraphRetriever, GraphRetrieverConfig
from knowledgebase.retrieval.retriever import FusionConfig, HybridRetriever
from knowledgebase.retrieval.vector_store import VectorStore

log = logging.getLogger(__name__)


class KnowledgeIndex:
    def __init__(
        self,
        chunks: list[Chunk],
        graph: KnowledgeGraph,
        embedder: Embedder | None,
        vectors: VectorStore | None,
        manifest: dict | None = None,
        graph_config: GraphRetrieverConfig | None = None,
        fusion: FusionConfig | None = None,
    ):
        self.chunks = chunks
        self.graph = graph
        self.embedder = embedder
        self.vectors = vectors
        self.manifest = manifest or {}
        self.chunk_pos = {c.id: i for i, c in enumerate(chunks)}
        self.bm25 = BM25Index().fit([c.search_text for c in chunks])
        self.graph_retriever = GraphRetriever(graph, chunks, graph_config)
        self.retriever = HybridRetriever(self, fusion)
        self._by_conversation: dict[str, list[int]] | None = None
        self._by_concept: dict[str, list[int]] | None = None

    # ------------------------------------------------------------------ lookups
    @property
    def by_conversation(self) -> dict[str, list[int]]:
        if self._by_conversation is None:
            d: dict[str, list[int]] = defaultdict(list)
            for i, c in enumerate(self.chunks):
                d[c.conversation_id].append(i)
            self._by_conversation = dict(d)
        return self._by_conversation

    @property
    def by_concept(self) -> dict[str, list[int]]:
        if self._by_concept is None:
            d: dict[str, list[int]] = defaultdict(list)
            for i, c in enumerate(self.chunks):
                for link in c.links:
                    d[link].append(i)
            self._by_concept = dict(d)
        return self._by_concept

    def search(self, query: str, k: int = 10, mode: str = "graph_hybrid"):
        return self.retriever.search(query, k=k, mode=mode)

    # ------------------------------------------------------------------ io
    @staticmethod
    def save(index_dir: Path, chunks: list[Chunk], graph: KnowledgeGraph, embeddings: np.ndarray | None,
             manifest: dict) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        with (index_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        if embeddings is not None:
            np.save(index_dir / "embeddings.npy", embeddings.astype(np.float32))
        elif (index_dir / "embeddings.npy").exists():
            (index_dir / "embeddings.npy").unlink()
        graph.save(index_dir / "graph.json")

        # Backwards-compatible exports used by the original pipeline/README.
        adjacency: dict[str, list[str]] = defaultdict(list)
        for e in graph.edges:
            adjacency[e["source"]].append(e["target"])
            adjacency[e["target"]].append(e["source"])
        (index_dir / "graph_edges.json").write_text(json.dumps(adjacency, indent=1), encoding="utf-8")
        (index_dir / "graph_relations.json").write_text(json.dumps(graph.relations, indent=1, ensure_ascii=False),
                                                        encoding="utf-8")
        entity_index: dict[str, list[str]] = defaultdict(list)
        for c in chunks:
            for link in c.links:
                entity_index[link].append(c.id)
        (index_dir / "entity_index.json").write_text(json.dumps(entity_index), encoding="utf-8")
        (index_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, index_dir: Path, embedder: Embedder | None | str = "auto", embed_model: str | None = None,
             graph_config: GraphRetrieverConfig | None = None, fusion: FusionConfig | None = None) -> "KnowledgeIndex":
        index_dir = Path(index_dir)
        manifest_path = index_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"No index found in {index_dir}. Run `kb build` first.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        with (index_dir / "chunks.jsonl").open("r", encoding="utf-8") as f:
            chunks = [Chunk.from_dict(json.loads(line)) for line in f if line.strip()]
        graph = KnowledgeGraph.load(index_dir / "graph.json")

        vectors = None
        emb_path = index_dir / "embeddings.npy"
        if isinstance(embedder, str):
            built_with = manifest.get("embedder")
            if emb_path.exists() and built_with:
                kind = "wordllama" if built_with.startswith("wordllama") else "sentence-transformers"
                model = embed_model or built_with.split(":", 1)[-1]
                embedder = build_embedder(kind, model)
            else:
                embedder = None
        if embedder is not None and emb_path.exists():
            if manifest.get("embedder") and embedder.name != manifest["embedder"]:
                log.warning("Index was built with %s but %s is loaded; dense retrieval disabled.",
                            manifest["embedder"], embedder.name)
                embedder = None
            else:
                vectors = VectorStore.load(emb_path)
        return cls(chunks, graph, embedder, vectors, manifest, graph_config, fusion)
