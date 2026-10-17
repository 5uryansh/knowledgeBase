from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from knowledgebase.text import strip_wikilinks


@dataclass(slots=True)
class Chunk:
    """One retrievable unit: a user prompt + the assistant's reply (or a piece of a long one)."""

    id: str
    conversation_id: str
    title: str
    platform: str
    date: str
    turn: int
    text: str  # markdown with [[wiki links]] (what the user sees)
    links: list[str] = field(default_factory=list)
    source: str = ""  # markdown file, relative to the vault
    hash: str = ""

    @property
    def plain(self) -> str:
        return strip_wikilinks(self.text)

    @property
    def search_text(self) -> str:
        """Text used for BM25/embeddings: contextual header + content."""
        return f"{self.title} ({self.date})\n{self.plain}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        return cls(**{k: data[k] for k in cls.__slots__ if k in data})
