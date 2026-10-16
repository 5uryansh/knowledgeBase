"""Topic extraction: YAKE keyphrases + spaCy noun chunks, lemmatised."""
from __future__ import annotations

import re
from collections import Counter

import yake

_BAD_CHARS = set("{}[]()/\\=:;<>@#$%^&*|~`\"")
_DETERMINERS = ("the ", "this ", "that ", "these ", "those ", "a ", "an ", "my ", "your ", "our ", "their ",
                "his ", "her ", "its ", "some ", "any ", "every ", "each ", "another ", "such ", "what ", "which ")


class TopicExtractor:
    def __init__(self, stop_words: frozenset[str]):
        self.stop_words = stop_words
        self.keyword_extractor = yake.KeywordExtractor(lan="en", n=3, dedupLim=0.3, top=8)

    def valid(self, topic: str) -> bool:
        t = topic.strip().lower()
        if t in self.stop_words or len(t) < 4 or len(t.split()) > 3:
            return False
        if re.search(r"\d", t) or not any(c.isalpha() for c in t) or any(c in _BAD_CHARS for c in t):
            return False
        words = t.split()
        if all(w in self.stop_words for w in words) or words[-1] in self.stop_words:
            return False
        return True

    @staticmethod
    def _strip_determiners(phrase: str) -> str:
        lowered = phrase.lower()
        changed = True
        while changed:
            changed = False
            for det in _DETERMINERS:
                if lowered.startswith(det):
                    phrase, lowered, changed = phrase[len(det):], lowered[len(det):], True
        return phrase.strip()

    def extract(self, text: str, docs) -> tuple[Counter, dict[str, set[str]]]:
        """Return (Counter[lemmatised topic -> score], {lemmatised topic -> surface forms})."""
        counter: Counter = Counter()
        lemma_of: dict[str, str] = {}
        surfaces: dict[str, set[str]] = {}
        for doc in docs:
            for chunk in doc.noun_chunks:
                if chunk.root.pos_ not in {"NOUN", "PROPN"}:
                    continue
                tokens = [t for t in chunk if not (t.is_stop or t.is_punct or t.pos_ in {"DET", "PRON", "NUM", "INTJ"})]
                while tokens and tokens[0].lower_ in self.stop_words:  # "hey caroline" -> "caroline"
                    tokens = tokens[1:]
                # Drop a leading person name glued to the phrase ("melanie pottery class" -> "pottery class")
                # but keep tech qualifiers ("Python script", "PyTorch model").
                while len(tokens) > 1 and tokens[0].ent_type_ == "PERSON":
                    tokens = tokens[1:]
                if not tokens or all(t.pos_ == "PROPN" for t in tokens) and len(tokens) > 1:
                    continue
                surface = doc.text[tokens[0].idx : tokens[-1].idx + len(tokens[-1].text)]
                lemma = " ".join([t.text.lower() for t in tokens[:-1]] + [tokens[-1].lemma_.lower()])
                if not self.valid(lemma):
                    continue
                lemma_of[surface.lower()] = lemma
                surfaces.setdefault(lemma, set()).add(surface)
                counter[lemma] += 1
        if len(text) > 200:
            for keyword, _score in self.keyword_extractor.extract_keywords(text[:200_000]):
                kw = self._strip_determiners(re.sub(r"\s+", " ", keyword.lower().strip()))
                words = kw.split()
                while words and words[0] in self.stop_words:
                    words = words[1:]
                kw = " ".join(words)
                kw = lemma_of.get(kw, kw)
                if self.valid(kw):
                    counter[kw] += 2
                    surfaces.setdefault(kw, set()).add(kw)
        return counter, surfaces
