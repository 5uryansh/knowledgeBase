"""Wiki-link injection: ``CUDA`` -> ``[[cuda|CUDA]]``.

Matching is a longest-first lookup of token n-grams in a dictionary of known
surface forms, so it scales to vocabularies of tens of thousands of concepts.
It only touches prose — never code blocks, inline code, URLs or existing links —
so it never produces nested links and is safe to re-run.
"""
from __future__ import annotations

import re
from typing import Iterable

from knowledgebase.enrichment.concepts import Concept

_PROTECTED = re.compile(r"```.*?```|`[^`\n]*`|\[\[[^\]]*\]\]|\[[^\]]*\]\([^)]*\)|https?://\S+", re.DOTALL)
# Underscores are word characters here so identifiers like esp_sleep_enable_timer_wakeup stay one token.
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+#.\-]*[A-Za-z0-9+#]|[A-Za-z0-9]")
MAX_NGRAM = 5


def surface_key(surface: str) -> tuple[str, ...]:
    return tuple(t.lower() for t in _TOKEN.findall(surface))


class Linker:
    def __init__(self, concepts: Iterable[Concept], min_surface_len: int = 3):
        self.lookup: dict[tuple[str, ...], str] = {}
        self.case_sensitive: dict[tuple[str, ...], set[str]] = {}
        for concept in sorted(concepts, key=lambda c: -c.score):  # higher-scoring concept wins collisions
            for surface in concept.surfaces | {concept.display}:
                s = surface.strip()
                key = surface_key(s)
                if not key or len(key) > MAX_NGRAM:
                    continue
                short = len(s) <= 3
                if len(s) < min_surface_len and not (s.isupper() and len(s) >= 2):
                    continue
                variants = [key]
                last = key[-1]
                if last.isalpha() and len(last) > 3:  # simple plural/singular variant of the head word
                    variants.append(key[:-1] + ((last[:-1],) if last.endswith("s") else (last + "s",)))
                for v in variants:
                    self.lookup.setdefault(v, concept.node)
                if short:  # "AI", "ML", "Go": require the exact casing to avoid linking ordinary words
                    self.case_sensitive.setdefault(key, set()).add(s)

    def _link_prose(self, text: str) -> str:
        if not self.lookup:
            return text
        tokens = list(_TOKEN.finditer(text))
        out, cursor, i = [], 0, 0
        while i < len(tokens):
            matched = False
            for n in range(min(MAX_NGRAM, len(tokens) - i), 0, -1):
                key = tuple(t.group(0).lower() for t in tokens[i:i + n])
                node = self.lookup.get(key)
                if node is None:
                    continue
                start, end = tokens[i].start(), tokens[i + n - 1].end()
                original = text[start:end]
                allowed = self.case_sensitive.get(key)
                if allowed is not None and original not in allowed:
                    continue
                # Only link contiguous phrases (no line breaks inside a multi-word match).
                if "\n" in original:
                    continue
                out.append(text[cursor:start])
                out.append(f"[[{node}|{original}]]")
                cursor, i, matched = end, i + n, True
                break
            if not matched:
                i += 1
        out.append(text[cursor:])
        return "".join(out)

    def link(self, text: str) -> str:
        out, last = [], 0
        for match in _PROTECTED.finditer(text):
            out.append(self._link_prose(text[last:match.start()]))
            out.append(match.group(0))
            last = match.end()
        out.append(self._link_prose(text[last:]))
        return "".join(out)
