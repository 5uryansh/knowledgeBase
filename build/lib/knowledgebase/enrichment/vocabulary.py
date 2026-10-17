"""Global concept vocabulary.

Concepts are *proposed* per conversation, but *selected* globally: something
mentioned once in each of five conversations is an important thread of your
thinking even though no single conversation makes it look important. Every
conversation is then linked against the whole vocabulary — that is what creates
the cross-conversation edges the graph exists for.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from knowledgebase.enrichment.concepts import Concept


@dataclass
class VocabEntry:
    node: str
    displays: Counter = field(default_factory=Counter)
    kinds: Counter = field(default_factory=Counter)
    labels: Counter = field(default_factory=Counter)
    surfaces: Counter = field(default_factory=Counter)
    score: float = 0.0
    conversations: int = 0

    def concept(self) -> Concept:
        kind = "entity" if self.kinds["entity"] >= self.kinds["topic"] else "topic"
        return Concept(
            node=self.node,
            display=self.displays.most_common(1)[0][0],
            kind=kind,
            label=self.labels.most_common(1)[0][0],
            score=self.score,
            surfaces=set(self.surfaces),
        )


class ConceptVocabulary:
    def __init__(self, entries: dict[str, VocabEntry]):
        self.entries = entries

    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, node: str) -> bool:
        return node in self.entries

    @classmethod
    def build(cls, concepts_by_conv: dict[str, list[Concept]], min_conversations: int = 1,
              min_score_single: float = 4.0, max_nodes: int = 50_000) -> "ConceptVocabulary":
        agg: dict[str, VocabEntry] = defaultdict(lambda: VocabEntry(node=""))
        for concepts in concepts_by_conv.values():
            for c in concepts:
                e = agg[c.node]
                e.node = c.node
                e.displays[c.display] += 1
                e.kinds[c.kind] += 1
                e.labels[c.label] += 1
                e.surfaces.update(c.surfaces)
                e.score += c.score
                e.conversations += 1
        kept = {
            node: e for node, e in agg.items()
            if e.conversations >= min_conversations or e.score >= min_score_single
        }
        if len(kept) > max_nodes:
            kept = dict(sorted(kept.items(), key=lambda kv: -kv[1].score)[:max_nodes])
        return cls(kept)

    def concepts(self) -> list[Concept]:
        return [e.concept() for e in self.entries.values()]
