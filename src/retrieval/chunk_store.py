from __future__ import annotations
import json
from pathlib import Path
from src.retrieval.chunk import Chunk


class ChunkStore:
    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}

    def build(self, chunks: list[Chunk]) -> None:
        self._chunks = {
            chunk.id: chunk
            for chunk in chunks
        }

    def get(self, chunk_id: str) -> Chunk:
        return self._chunks[chunk_id]

    def get_many(self, chunk_ids: list[str] | set[str]) -> list[Chunk]:
        return [
            self._chunks[chunk_id]
            for chunk_id in chunk_ids
            if chunk_id in self._chunks
        ]

    def save(self, path: Path) -> None:
        with path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    cid: chunk.to_dict()
                    for cid, chunk in self._chunks.items()
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, path: Path) -> "ChunkStore":
        store = cls()

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        store._chunks = {
            cid: Chunk.from_dict(chunk)
            for cid, chunk in data.items()
        }

        return store