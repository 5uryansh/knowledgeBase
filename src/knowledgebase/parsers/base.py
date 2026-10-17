"""Platform-agnostic conversation model.

Every exporter (ChatGPT, Claude, Gemini, Claude Code, ...) is normalised into
:class:`Conversation` objects. Adding a new provider only requires a parser that
yields these — everything downstream (enrichment, graph, retrieval) is shared.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterator, Protocol


@dataclass
class Message:
    role: str  # "user" | "assistant"
    text: str
    timestamp: str = ""


@dataclass
class Conversation:
    platform: str
    id: str
    title: str
    created_at: datetime | None
    messages: list[Message] = field(default_factory=list)
    project: str | None = None

    @property
    def date(self) -> str:
        return self.created_at.strftime("%Y-%m-%d") if self.created_at else "unknown-date"

    @property
    def content_hash(self) -> str:
        h = hashlib.sha256()
        for m in self.messages:
            h.update(m.role.encode())
            h.update(m.text.encode("utf-8"))
        return h.hexdigest()[:8]

    @property
    def uid(self) -> str:
        """Stable id across re-imports (platform + native conversation id)."""
        return hashlib.sha1(f"{self.platform}:{self.id}".encode()).hexdigest()[:10]


class ConversationParser(Protocol):
    platform: str

    def parse(self) -> Iterator[Conversation]:  # pragma: no cover - protocol
        ...


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt


def parse_epoch(value: float | int | None) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    except (TypeError, ValueError, OverflowError, OSError):
        return None
