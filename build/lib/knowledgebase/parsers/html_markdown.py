"""Minimal HTML → Markdown conversion for Gemini Takeout responses."""
from __future__ import annotations

import re
from html.parser import HTMLParser

_BAD_STRINGS = {"text", "label", "['text', 'label']", '["text", "label"]'}


class HTMLToMarkdownParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._in_pre = False

    def handle_starttag(self, tag, attrs):
        if tag in ("p", "div"):
            self.parts.append("\n\n")
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            # Never emit "## " — that level is reserved for Prompt/Response turns.
            self.parts.append("\n\n**")
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag == "br":
            self.parts.append("\n")
        elif tag in ("strong", "b"):
            self.parts.append("**")
        elif tag in ("em", "i"):
            self.parts.append("*")
        elif tag == "pre":
            self._in_pre = True
            self.parts.append("\n\n```\n")
        elif tag == "code" and not self._in_pre:
            self.parts.append("`")

    def handle_endtag(self, tag):
        if tag in ("strong", "b"):
            self.parts.append("**")
        elif tag in ("em", "i"):
            self.parts.append("*")
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.parts.append("**\n\n")
        elif tag == "pre":
            self._in_pre = False
            self.parts.append("\n```\n\n")
        elif tag == "code" and not self._in_pre:
            self.parts.append("`")
        elif tag in ("p", "ul", "ol"):
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_pre:
            self.parts.append(data)
            return
        if data.strip().lower() in _BAD_STRINGS:
            return
        self.parts.append(re.sub(r"\s+", " ", data))

    def get_markdown(self) -> str:
        text = "".join(self.parts)
        text = re.sub(r"[ \t]+\n", "\n", text)
        return re.sub(r"\n{3,}", "\n\n", text).strip()


def html_to_markdown(html: str) -> str:
    parser = HTMLToMarkdownParser()
    parser.feed(html or "")
    parser.close()
    return parser.get_markdown()
