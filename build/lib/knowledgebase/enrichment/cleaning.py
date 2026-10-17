"""Text cleaning before NLP: drop code, URLs, markup and chat scaffolding."""
from __future__ import annotations

import re

_CODE_BLOCK = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`\n]*`")
_URL = re.compile(r"https?://\S+")
_WIKILINK = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_SCAFFOLD = re.compile(r"(?m)^#{1,6}\s*(Prompt|Response|Timestamp)\s*$")
_ISO_TS = re.compile(r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b")
_MD_SYMBOLS = re.compile(r"[#>*_|]+")
_SPEAKER = re.compile(r"(?m)^[A-Z][\w .'-]{0,30}:\s+")  # "Caroline: ..." transcript speaker labels


def clean_for_nlp(text: str) -> str:
    text = _CODE_BLOCK.sub(" ", text)
    text = _INLINE_CODE.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _WIKILINK.sub(lambda m: m.group(2) or m.group(1), text)
    text = _MD_LINK.sub(r"\1", text)
    text = _SCAFFOLD.sub(" ", text)
    text = _SPEAKER.sub("", text)
    text = _ISO_TS.sub(" ", text)
    text = _MD_SYMBOLS.sub(" ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()
