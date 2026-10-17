"""How *specific* a concept is — "vLLM" or "partial refresh" vs "model" or "update".

Generic words are legitimate nodes for retrieval (IDF already discounts them),
but they make a terrible first impression in the graph explorer and produce
boring "discoveries". Specificity scales the display importance and filters
discovery endpoints.

Uses `wordfreq` (word frequencies across many corpora) when installed, otherwise
a built-in list of generic words.
"""
from __future__ import annotations

import re
from functools import lru_cache

try:  # optional, ~57 MB of frequency data
    from wordfreq import zipf_frequency as _zipf
except ImportError:  # pragma: no cover - depends on environment
    _zipf = None

# Fallback: common nouns/verbs that show up in technical and casual chats.
_GENERIC = frozenset("""
model models graph concept client node nodes query queries memory update updates return returns weight weights
compute read write second seconds point points edge edges note notes default defaults request requests server
servers response result value values item items list lists name names user users step steps way ways thing things
part parts time times day days week weeks year years number numbers size type types line lines page pages word words
question questions answer answers example examples problem problems issue issues case cases change changes test
tests run runs build builds setup start end level levels set sets option options feature features version versions
data file files text code function functions method methods class classes object objects field fields table tables
key keys output input source target state states process work job jobs task tasks group groups team life love family
friend friends kid kids home place world idea ideas story stories art music book books movie movies game games photo
picture pictures image images video videos call calls message messages chat link links site website app apps tool
tools system systems service services support help stuff lot bit kind sort man woman people person child children
idea plan plans goal goals reason result quality speed cost price memory performance information detail details
action actions training learning design support use usage approach practice experience feeling feelings sense care
""".split())

_TOKEN = re.compile(r"[a-z0-9][a-z0-9+#.\-]*")


@lru_cache(maxsize=200_000)
def word_zipf(word: str) -> float:
    if _zipf is not None:
        return _zipf(word, "en")
    if word in _GENERIC:
        return 5.0
    return 2.5 if word.isalpha() else 0.0


def specificity(label: str, kind: str = "topic", type_: str = "") -> float:
    """1.0 = very specific (rare term / named thing), ~0.15 = very generic word."""
    words = _TOKEN.findall(label.lower())
    if not words:
        return 0.0
    per_word = [min(1.0, max(0.15, (5.5 - word_zipf(w)) / 3.0)) for w in words]
    score = sum(per_word) / len(per_word) + 0.25 * (len(words) - 1)
    if kind == "entity" and type_ not in ("topic", ""):
        score += 0.15
    if any(c.isdigit() for c in label) or (label.isupper() and len(label) >= 2):
        score += 0.2  # ESP32, GPT-4, CUDA
    return round(min(1.0, score), 4)
