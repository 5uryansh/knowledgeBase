"""Named-entity extraction backends.

* ``GlinerEntityExtractor`` — zero-shot NER with GLiNER (best quality; needs the
  model from Hugging Face the first time).
* ``SpacyEntityExtractor`` — spaCy NER + technical-term patterns. Fully offline,
  used automatically when GLiNER is not installed or the model can't be fetched.

Both return raw ``(surface, label, weight)`` hits; canonicalisation and
filtering happen in :mod:`knowledgebase.enrichment.concepts`.
"""
from __future__ import annotations

import logging
import re
from typing import Iterable, Protocol

log = logging.getLogger(__name__)

# Technical terms spaCy's news-trained NER tends to miss.
TECH_PATTERNS = [
    re.compile(r"\b(?:GPT|Gemma|Llama|BERT|Claude|Gemini|Mistral|Qwen|Phi|T5)[-‑ ]?\d[\w.]*\b"),  # model names + version
    re.compile(r"\b[A-Z][a-z]+(?:[A-Z][a-z0-9]*)+\b"),  # CamelCase: PyTorch, FastAPI, GraphRAG
    re.compile(r"\b[A-Z]{2,6}\d{0,4}\b"),  # acronyms: CUDA, FAISS, ESP32, RAG
    re.compile(r"\b[A-Za-z][\w-]*\.(?:js|ts|py|md|io|ai|rs)\b"),  # Node.js, obsidian.md
]
_ACRONYM_BLOCKLIST = {
    "I", "OK", "AM", "PM", "TV", "ID", "OS", "UI", "UX", "FAQ", "TL", "DR", "TLDR", "LOL", "OMG", "BTW",
    "IMO", "ASAP", "FYI", "USA", "UK", "EU", "AND", "THE", "FOR", "NOT", "YES", "NO", "IT", "IS",
    "KB", "MB", "GB", "TB", "PB", "MS", "US", "NS", "HZ", "MHZ", "GHZ", "FPS", "RPM", "ETA", "PS", "PPS",
}

SPACY_LABELS = {
    "PERSON": "person",
    "ORG": "organization",
    "GPE": "location",
    "LOC": "location",
    "FAC": "location",
    "PRODUCT": "product",
    "EVENT": "event",
    "WORK_OF_ART": "work",
    "NORP": "group",
    "LANGUAGE": "language",
    "LAW": "concept",
}


class EntityExtractor(Protocol):
    name: str

    def extract(self, texts: list[str]) -> list[tuple[str, str, float]]:  # pragma: no cover
        ...


def tech_term_hits(text: str) -> Iterable[tuple[str, str, float]]:
    for pattern in TECH_PATTERNS:
        for match in pattern.findall(text):
            if match in _ACRONYM_BLOCKLIST or match.isdigit():
                continue
            yield match, "technology", 1.0


class SpacyEntityExtractor:
    name = "spacy"

    def __init__(self, nlp):
        self.nlp = nlp

    def extract_from_docs(self, docs, texts: list[str]) -> list[tuple[str, str, float]]:
        hits: list[tuple[str, str, float]] = []
        for doc in docs:
            for ent in doc.ents:
                label = SPACY_LABELS.get(ent.label_)
                if not label:
                    continue
                surface = ent.text.strip(" .,:;!?'\"()[]")
                surface = re.sub(r"^(the|a|an)\s+", "", surface, flags=re.I)
                surface = re.sub(r"'s$", "", surface)
                if surface:
                    hits.append((surface, label, 2.0))
        for text in texts:
            hits.extend(tech_term_hits(text))
        return hits

    def extract(self, texts: list[str]) -> list[tuple[str, str, float]]:
        return self.extract_from_docs(self.nlp.pipe(texts, batch_size=32), texts)


class GlinerEntityExtractor:
    name = "gliner"

    def __init__(self, model_name: str, labels: list[str]):
        from gliner import GLiNER  # imported lazily: heavy, optional dependency

        try:
            self.model = GLiNER.from_pretrained(model_name, local_files_only=True)
        except Exception:  # noqa: BLE001 - not cached yet
            log.info("Downloading GLiNER model %s ...", model_name)
            self.model = GLiNER.from_pretrained(model_name)
        self.labels = labels

    @staticmethod
    def _windows(text: str, size: int = 1200) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        windows, current = [], ""
        for sentence in sentences:
            if len(current) + len(sentence) < size:
                current = f"{current} {sentence}" if current else sentence
            else:
                if current:
                    windows.append(current)
                current = sentence[:size]
        if current:
            windows.append(current)
        return windows

    def extract(self, texts: list[str]) -> list[tuple[str, str, float]]:
        hits: list[tuple[str, str, float]] = []
        for text in texts:
            for window in self._windows(text):
                for pred in self.model.predict_entities(window, self.labels):
                    hits.append((pred["text"].strip(), pred.get("label", "concept"), 2.0))
            hits.extend(tech_term_hits(text))
        return hits


def build_entity_extractor(kind: str, nlp, gliner_model: str, labels: list[str]):
    """Return the requested backend, falling back to spaCy when GLiNER is unavailable."""
    if kind in ("auto", "gliner"):
        try:
            return GlinerEntityExtractor(gliner_model, labels)
        except Exception as exc:  # noqa: BLE001
            if kind == "gliner":
                raise
            log.warning("GLiNER unavailable (%s); falling back to spaCy NER.", exc.__class__.__name__)
    return SpacyEntityExtractor(nlp)
