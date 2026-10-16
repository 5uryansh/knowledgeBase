"""Concept extraction = entities + topics, canonicalised into graph node ids."""
from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from knowledgebase.config import Settings, aliases, entity_labels, stop_words
from knowledgebase.enrichment.cleaning import clean_for_nlp
from knowledgebase.enrichment.entities import SpacyEntityExtractor, build_entity_extractor
from knowledgebase.enrichment.topics import TopicExtractor
from knowledgebase.parsers.base import Conversation
from knowledgebase.text import normalize_node_name

log = logging.getLogger(__name__)

_RELATION_VERBS = {"uses", "causes", "handles", "using", "based", "improves", "results", "depends", "supports"}


@dataclass
class Concept:
    node: str
    display: str
    kind: str  # "entity" | "topic"
    label: str
    score: float
    surfaces: set[str] = field(default_factory=set)

    def to_dict(self) -> dict:
        return {
            "node": self.node,
            "display": self.display,
            "kind": self.kind,
            "label": self.label,
            "score": self.score,
            "surfaces": sorted(self.surfaces),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Concept":
        return cls(d["node"], d["display"], d["kind"], d["label"], d["score"], set(d.get("surfaces", [])))


def load_spacy(model: str):
    import spacy

    try:
        nlp = spacy.load(model)
    except OSError as exc:  # pragma: no cover - environment specific
        raise SystemExit(
            f"spaCy model '{model}' is not installed. Run: python -m spacy download {model}"
        ) from exc
    nlp.max_length = 2_000_000
    return nlp


class ConceptExtractor:
    """Turns a conversation into a ranked list of canonical concepts."""

    VERSION = "8"

    def __init__(self, settings: Settings, nlp=None):
        self.settings = settings
        self.nlp = nlp or load_spacy(settings.spacy_model)
        self.stop_words = stop_words(settings)
        self.aliases = aliases(settings)
        self.entity_extractor = build_entity_extractor(
            settings.entity_extractor, self.nlp, settings.gliner_model, entity_labels(settings)
        )
        self.topic_extractor = TopicExtractor(self.stop_words)
        self.max_entities = 80
        self.max_topics = 80
        # Per-conversation candidate thresholds are deliberately low: the global
        # vocabulary step (see vocabulary.py) decides what really becomes a node.
        self.min_topic_count = 1
        self.min_entity_score = 1.0

    @property
    def signature(self) -> str:
        return f"concepts-v{self.VERSION}-{self.entity_extractor.name}"

    # ------------------------------------------------------------------ helpers
    def canonical(self, surface: str) -> tuple[str, str]:
        surface = re.sub(r"\s+", " ", surface).strip()
        surface = re.sub(r"\b([A-Z]{2,})s\b", r"\1", surface)  # GPUs -> GPU, LLMs -> LLM
        display = self.aliases.get(surface.lower(), surface)
        return normalize_node_name(display), display

    def _valid_entity(self, surface: str) -> bool:
        s = surface.strip()
        is_acronym = s.isupper() and 2 <= len(s) <= 6
        if (len(s) < 3 and not is_acronym) or len(s.split()) > 4 or len(s) > 60:
            return False
        low = s.lower()
        if low in self.stop_words or low in _RELATION_VERBS or s.isdigit():
            return False
        if not any(c.isalpha() for c in s):
            return False
        if all(w in self.stop_words for w in low.split()):
            return False
        return True

    # ------------------------------------------------------------------ main API
    def extract(self, conversation: Conversation) -> list[Concept]:
        texts = [clean_for_nlp(m.text) for m in conversation.messages]
        texts = [t for t in texts if t]
        if not texts:
            return []
        docs = list(self.nlp.pipe(texts, batch_size=32))
        # How each lowercase word is used in this conversation: lets us reject
        # "Glad"/"Perfect"/"Thank" that NER mislabels as names in casual chat.
        pos_of: dict[str, Counter] = defaultdict(Counter)
        for doc in docs:
            for tok in doc:
                pos_of[tok.lower_][tok.pos_] += 1

        if isinstance(self.entity_extractor, SpacyEntityExtractor):
            hits = self.entity_extractor.extract_from_docs(docs, texts)
        else:
            hits = self.entity_extractor.extract(texts)

        ent_score: Counter = Counter()
        ent_labels: dict[str, Counter] = defaultdict(Counter)
        ent_surfaces: dict[str, Counter] = defaultdict(Counter)
        ent_display: dict[str, str] = {}
        for surface, label, weight in hits:
            if not self._valid_entity(surface):
                continue
            if " " not in surface.strip() and not surface.isupper():
                usage = pos_of.get(surface.lower())
                if usage:
                    nounish = usage["NOUN"] + usage["PROPN"]
                    if nounish / sum(usage.values()) < 0.5:
                        continue
            node, display = self.canonical(surface)
            if not node:
                continue
            ent_score[node] += weight
            ent_labels[node][label] += 1
            ent_surfaces[node][surface] += 1
            ent_display.setdefault(node, display)

        concepts: dict[str, Concept] = {}
        for node, score in ent_score.most_common():
            if score < self.min_entity_score:
                continue
            surfaces = ent_surfaces[node]
            display = ent_display[node]
            if display.lower() not in self.aliases:  # prefer the most frequent surface spelling
                display = surfaces.most_common(1)[0][0]
            concepts[node] = Concept(
                node=node,
                display=display,
                kind="entity",
                label=ent_labels[node].most_common(1)[0][0],
                score=float(score),
                surfaces=set(surfaces),
            )
            if len(concepts) >= self.max_entities:
                break

        full_text = "\n\n".join(texts)
        n_words = len(full_text.split())
        # Long conversations need more evidence before a phrase counts as a topic.
        min_topic = max(self.min_topic_count, min(6, n_words // 1500))
        topic_counts, topic_surfaces = self.topic_extractor.extract(full_text, docs)
        added = 0
        for phrase, count in topic_counts.most_common():
            if count < min_topic or added >= self.max_topics:
                break
            node, display = self.canonical(phrase)
            if not node:
                continue
            surfaces = topic_surfaces.get(phrase, {phrase})
            if node in concepts:
                concepts[node].score += count
                concepts[node].surfaces |= surfaces
                continue
            if display == phrase:
                display = phrase[:1].upper() + phrase[1:]
            concepts[node] = Concept(node, display, "topic", "topic", float(count), set(surfaces) | {phrase})
            added += 1

        return sorted(concepts.values(), key=lambda c: -c.score)
