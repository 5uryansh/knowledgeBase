from __future__ import annotations
from pathlib import Path
from src.retrieval.chunk_store import ChunkStore
from src.retrieval.chunker import Chunker
from src.retrieval.embedder import Embedder
from src.retrieval.entity_index import EntityIndex
from src.retrieval.vector_store import VectorStore


class Indexer:
    def __init__(self, chunker: Chunker, embedder: Embedder) -> None:
        self._chunker = chunker
        self._embedder = embedder

    def build(self, markdown_files: list[Path], output_dir: Path) -> None:
        chunks = []

        for file in markdown_files:
            chunks.extend(self._chunker.chunk(file))
        print(f"Markdown files: {len(markdown_files)}")
        print(f"Chunks: {len(chunks)}")

        print("Generating embeddings...")
        if not chunks:
            raise ValueError("No chunks were generated.")
        embeddings = self._embedder.embed(chunks)
        print(f"Generated {len(embeddings)} embeddings")

        vector_store = VectorStore(dimension=len(embeddings[0]))
        vector_store.add([chunk.id for chunk in chunks], embeddings)

        chunk_store = ChunkStore()
        chunk_store.build(chunks)

        entity_index = EntityIndex()
        entity_index.build(chunks)

        vector_store.save(output_dir / "embeddings.faiss", output_dir / "chunk_ids.txt")
        chunk_store.save(output_dir / "chunks.json")
        entity_index.save(output_dir / "entity_index.json")