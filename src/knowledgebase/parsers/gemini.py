"""Gemini parser (Google Takeout → My Activity → Gemini Apps → ``MyActivity.json``).

Takeout has no conversation ids, so activity items from the same day are grouped
into one conversation.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterator

from knowledgebase.parsers.base import Conversation, Message, parse_iso
from knowledgebase.parsers.html_markdown import html_to_markdown

_PROMPT_PREFIXES = ("Prompted ", "Asked ")


class GeminiChatParser:
    platform = "gemini"

    def __init__(self, input_file: str | Path):
        self.input_file = Path(input_file)

    def parse(self) -> Iterator[Conversation]:
        data = json.loads(self.input_file.read_text(encoding="utf-8"))
        by_day: dict[str, list[dict]] = defaultdict(list)
        for item in sorted(data, key=lambda x: x.get("time", "")):
            dt = parse_iso(item.get("time"))
            by_day[dt.strftime("%Y-%m-%d") if dt else "unknown-date"].append(item)

        for day, items in by_day.items():
            messages: list[Message] = []
            for item in items:
                prompt = item.get("title") or ""
                for prefix in _PROMPT_PREFIXES:
                    if prompt.startswith(prefix):
                        prompt = prompt[len(prefix):]
                response = "\n\n".join(
                    html_to_markdown(h.get("html", "")) for h in item.get("safeHtmlItem") or []
                ).strip()
                if prompt.strip():
                    messages.append(Message("user", prompt.strip(), item.get("time", "")))
                if response:
                    messages.append(Message("assistant", response, item.get("time", "")))
            if not messages:
                continue
            first_prompt = next((m.text for m in messages if m.role == "user"), "Gemini")
            yield Conversation(
                platform=self.platform,
                id=f"gemini-{day}",
                title=f"Gemini — {first_prompt[:60]}" + (f" (+{len(items) - 1} more)" if len(items) > 1 else ""),
                created_at=parse_iso(items[0].get("time")),
                messages=messages,
            )
