from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
from src.retrieval.chunk import Chunk


class EntityIndex:
    def __init__(self) -> None:
        self._index: dict[str, list[str]] = defaultdict(list)

    def build(self, chunks: list[Chunk]) -> None:
        self._index.clear()
        for chunk in chunks:
            for entity in chunk.links:
                self._index[entity].append(chunk.id)

    def lookup(self, entities: list[str] | set[str]) -> set[str]:
        chunk_ids: set[str] = set()
        for entity in entities:
            chunk_ids.update(self._index.get(entity, []))
        return chunk_ids

    def save(self, path: Path) -> None:
        with path.open("w", encoding="utf-8") as f:
            json.dump(self._index, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "EntityIndex":
        index = cls()
        with path.open("r", encoding="utf-8") as f:
            index._index = json.load(f)
        return index