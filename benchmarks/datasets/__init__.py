"""Benchmark datasets converted into knowledgeBase's own input formats."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Query:
    id: str
    text: str
    category: str
    answer: str = ""
    # Each evidence item is satisfied if ANY of its chunk ids is retrieved.
    evidence: list[set[str]] = field(default_factory=list)
    evidence_sessions: set[str] = field(default_factory=set)


@dataclass
class Corpus:
    """One independent knowledge base (e.g. one LoCoMo user) + the questions asked about it."""

    id: str
    export_path: str  # a chat export in a format one of our parsers understands
    platform: str
    queries: list[Query] = field(default_factory=list)
    # message locator -> (conversation native id, message index); resolved to chunk ids after building
    evidence_locators: dict[str, tuple[str, int]] = field(default_factory=dict)
