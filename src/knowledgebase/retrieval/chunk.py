from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Chunk:
    id: str
    text: str
    source: Path
    links: list[str]
    hash: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source"] = str(self.source)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        return cls(
            id=data["id"],
            text=data["text"],
            source=Path(data["source"]),
            links=data["links"],
            hash=data["hash"],
        )