"""Typed relation extraction: ``[[cuda]] accelerates [[pytorch]]`` -> (cuda, OPTIMIZES, pytorch).

Instead of linking *every* pair of nodes in a sentence that contains a trigger
(which produced lots of junk), only the nearest linked node before the trigger
and the nearest linked node after it are related. Each relation keeps a short
evidence sentence so the UI/MCP can show *why* two things are connected.
"""
from __future__ import annotations

import re
from collections import defaultdict

from knowledgebase.graph.relation_patterns import RELATION_PATTERNS
from knowledgebase.text import WIKILINK, strip_wikilinks

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n")
_TRIGGERS = sorted(
    ((trigger, relation) for relation, triggers in RELATION_PATTERNS.items() for trigger in triggers),
    key=lambda t: -len(t[0]),
)
_TRIGGER_RE = re.compile(r"\b(" + "|".join(re.escape(t) for t, _ in _TRIGGERS) + r")\b", re.IGNORECASE)
_TRIGGER_TO_REL = {t.lower(): r for t, r in _TRIGGERS}


class RelationExtractor:
    def __init__(self, max_evidence: int = 3):
        self.relations: dict[tuple[str, str, str], dict] = defaultdict(
            lambda: {"weight": 0, "conversations": set(), "evidence": []}
        )
        self.max_evidence = max_evidence

    def process(self, text: str, conversation_id: str) -> None:
        for sentence in _SENTENCE_SPLIT.split(text):
            links = [(m.start(), m.end(), m.group(1).strip().lower()) for m in WIKILINK.finditer(sentence)]
            if len(links) < 2:
                continue
            for trig in _TRIGGER_RE.finditer(sentence):
                before = [l for l in links if l[1] <= trig.start()]
                after = [l for l in links if l[0] >= trig.end()]
                if not before or not after:
                    continue
                source, target = before[-1][2], after[0][2]
                # The trigger must sit between the two nodes, close to both.
                if source == target or trig.start() - before[-1][1] > 60 or after[0][0] - trig.end() > 60:
                    continue
                relation = _TRIGGER_TO_REL[trig.group(1).lower()]
                entry = self.relations[(source, relation, target)]
                entry["weight"] += 1
                entry["conversations"].add(conversation_id)
                if len(entry["evidence"]) < self.max_evidence:
                    evidence = strip_wikilinks(sentence).strip()
                    if evidence not in entry["evidence"]:
                        entry["evidence"].append(evidence[:280])

    def export(self) -> list[dict]:
        return [
            {
                "source": s,
                "relation": r,
                "target": t,
                "weight": v["weight"],
                "conversations": sorted(v["conversations"]),
                "evidence": v["evidence"],
            }
            for (s, r, t), v in sorted(self.relations.items(), key=lambda kv: -kv[1]["weight"])
        ]
