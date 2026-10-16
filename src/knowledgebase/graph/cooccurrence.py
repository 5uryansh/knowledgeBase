"""Weighted co-occurrence graph built from linked chunks.

Raw co-occurrence alone is dominated by hub nodes (the user's own name, "python",
"model" ...) that co-occur with everything. Each edge therefore also gets a
conversation-level NPMI score (normalised pointwise mutual information, in
[-1, 1]): high when two concepts appear together far more than chance would
predict. ``strength`` combines both and is what retrieval and the UI use.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations

from knowledgebase.text import WIKILINK

CONTEXT_WEIGHTS = {"sentence": 5.0, "paragraph": 2.0, "conversation": 1.0}
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n")


def links_in(text: str) -> list[str]:
    return sorted({m.group(1).strip().lower() for m in WIKILINK.finditer(text)})


@dataclass(slots=True)
class EdgeStats:
    sentence: int = 0
    paragraph: int = 0
    conversations: int = 0

    @property
    def weight(self) -> float:
        return (
            CONTEXT_WEIGHTS["sentence"] * self.sentence
            + CONTEXT_WEIGHTS["paragraph"] * self.paragraph
            + CONTEXT_WEIGHTS["conversation"] * self.conversations
        )


class CooccurrenceBuilder:
    def __init__(self, max_links_per_conversation: int = 25):
        self.edges: dict[tuple[str, str], EdgeStats] = defaultdict(EdgeStats)
        self.node_conversations: dict[str, set[str]] = defaultdict(set)
        self.node_mentions: Counter = Counter()
        self.conversations: set[str] = set()
        self.max_links = max_links_per_conversation

    @staticmethod
    def _key(a: str, b: str) -> tuple[str, str]:
        return (a, b) if a < b else (b, a)

    def add_conversation(self, conversation_id: str, texts: list[str]) -> None:
        """``texts`` are the linked messages (or chunks) of one conversation."""
        self.conversations.add(conversation_id)
        conv_nodes: Counter = Counter()
        conv_pairs: set[tuple[str, str]] = set()
        for text in texts:
            for m in WIKILINK.finditer(text):
                conv_nodes[m.group(1).strip().lower()] += 1
            for paragraph in text.split("\n\n"):
                p_nodes = links_in(paragraph)
                if len(p_nodes) >= 2:
                    for a, b in combinations(p_nodes, 2):  # p_nodes is sorted -> (a, b) is canonical
                        self.edges[(a, b)].paragraph += 1
                        conv_pairs.add((a, b))
                    for sentence in _SENTENCE_SPLIT.split(paragraph):
                        s_nodes = links_in(sentence)
                        for a, b in combinations(s_nodes, 2):
                            self.edges[(a, b)].sentence += 1
        self.node_mentions.update(conv_nodes)
        for node in conv_nodes:
            self.node_conversations[node].add(conversation_id)
        # Conversation-level co-occurrence: every locally co-occurring pair, plus all
        # pairs among the conversation's most central concepts (all-pairs over every
        # concept would be quadratic and mostly noise).
        top = sorted(n for n, _ in conv_nodes.most_common(self.max_links))
        conv_pairs.update(combinations(top, 2))
        for pair in conv_pairs:
            self.edges[pair].conversations += 1

    def npmi(self, a: str, b: str, stats: EdgeStats) -> float:
        n = max(len(self.conversations), 1)
        c_ab = stats.conversations
        if c_ab == 0:
            return 0.0
        p_ab = c_ab / n
        p_a = len(self.node_conversations.get(a, ())) / n
        p_b = len(self.node_conversations.get(b, ())) / n
        if p_ab >= 1.0 or p_a == 0 or p_b == 0:
            return 0.0
        return math.log(p_ab / (p_a * p_b)) / -math.log(p_ab)

    def export(self, min_conversations_without_local: int = 2) -> list[dict]:
        edges = []
        for (a, b), stats in self.edges.items():
            local = stats.sentence + stats.paragraph
            if local == 0 and stats.conversations < min_conversations_without_local:
                continue
            npmi = self.npmi(a, b, stats)
            strength = math.log1p(stats.weight) * max(npmi, 0.0)
            contexts = [c for c, v in (("sentence", stats.sentence), ("paragraph", stats.paragraph),
                                       ("conversation", stats.conversations)) if v]
            edges.append({
                "source": a,
                "target": b,
                "weight": round(stats.weight, 3),
                "sentence": stats.sentence,
                "paragraph": stats.paragraph,
                "conversations": stats.conversations,
                "npmi": round(npmi, 4),
                "strength": round(strength, 4),
                "contexts": contexts,
            })
        return edges
