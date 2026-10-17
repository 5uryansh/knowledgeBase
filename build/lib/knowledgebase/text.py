"""Small text helpers shared across the pipeline."""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

_WS = re.compile(r"\s+")
_NON_WORD = re.compile(r"[^a-z0-9_]")
WIKILINK = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")
_TOKEN = re.compile(r"[a-z0-9][a-z0-9+#._-]*[a-z0-9+#]|[a-z0-9]")

# Compact English stopword list for lexical matching (BM25, entity linking).
STOPWORDS = frozenset(
    """a about above after again against all am an and any are aren't as at be because been before
    being below between both but by can can't cannot could couldn't did didn't do does doesn't doing
    don't down during each few for from further had hadn't has hasn't have haven't having he he'd he'll
    he's her here here's hers herself him himself his how how's i i'd i'll i'm i've if in into is isn't
    it it's its itself let's me more most mustn't my myself no nor not of off on once only or other ought
    our ours ourselves out over own same shan't she she'd she'll she's should shouldn't so some such than
    that that's the their theirs them themselves then there there's these they they'd they'll they're
    they've this those through to too under until up very was wasn't we we'd we'll we're we've were
    weren't what what's when when's where where's which while who who's whom why why's with won't would
    wouldn't you you'd you'll you're you've your yours yourself yourselves also just like get got really
    yeah yes hey hi oh ok okay thanks thank sure well much many one would could did""".split()
)


def normalize_node_name(text: str) -> str:
    """Canonical node id: ascii, lowercase, words joined by underscores."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.strip().lower()
    text = text.replace("+", "plus").replace("#", "sharp") if re.fullmatch(r"[a-z]+[+#]+", text) else text
    text = _WS.sub("_", text)
    text = _NON_WORD.sub("", text)
    return re.sub(r"_+", "_", text).strip("_")


def sanitize_filename(text: str) -> str:
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", text).strip()


def slugify(text: str, max_len: int = 48) -> str:
    slug = normalize_node_name(text).replace("_", "-")
    return slug[:max_len].strip("-") or "untitled"


def strip_wikilinks(text: str) -> str:
    """[[node|Display]] -> Display, [[node]] -> node."""
    return WIKILINK.sub(lambda m: m.group(2) or m.group(1), text)


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    tokens = _TOKEN.findall(text.lower())
    if drop_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return [stem(t) for t in tokens]


def stem(token: str) -> str:
    """Very light suffix stripping — enough to match plurals and simple verb forms."""
    if len(token) <= 4 or not token.isalpha():
        return token
    for suffix, repl in (("ies", "y"), ("sses", "ss"), ("ing", ""), ("edly", ""), ("ed", ""), ("es", ""), ("s", "")):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            if suffix == "s" and token.endswith(("ss", "us", "is")):
                return token
            if suffix == "es" and not token.endswith(("ches", "shes", "xes", "zes", "sses")):
                return token[:-1]
            return token[: len(token) - len(suffix)] + repl
    return token


def find_markdown_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.md")) if root.exists() else []
