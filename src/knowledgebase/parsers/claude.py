"""Claude.ai export parser (``conversations.json`` from Settings → Privacy → Export data)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from knowledgebase.parsers.base import Conversation, Message, parse_iso


class ClaudeChatParser:
    platform = "claude"

    def __init__(self, input_file: str | Path, include_thinking: bool = False):
        self.input_file = Path(input_file)
        self.include_thinking = include_thinking

    def _extract_message_text(self, message: dict) -> str:
        parts: list[str] = []
        for content in message.get("content") or []:
            ctype = content.get("type")
            if ctype == "text" and (content.get("text") or "").strip():
                parts.append(content["text"])
            elif ctype == "thinking" and self.include_thinking and (content.get("thinking") or "").strip():
                parts.append(f"> Claude Thinking\n>\n> {content['thinking']}")
        if not parts and (message.get("text") or "").strip():
            parts.append(message["text"])  # older exports
        return "\n\n".join(parts).strip()

    def parse(self) -> Iterator[Conversation]:
        data = json.loads(self.input_file.read_text(encoding="utf-8"))
        for conv in sorted(data, key=lambda c: c.get("created_at") or ""):
            messages: list[Message] = []
            for message in sorted(conv.get("chat_messages") or [], key=lambda m: m.get("created_at") or ""):
                text = self._extract_message_text(message)
                if not text:
                    continue
                role = "user" if message.get("sender") == "human" else "assistant"
                messages.append(Message(role=role, text=text, timestamp=message.get("created_at") or ""))
            if not messages:
                continue
            yield Conversation(
                platform=self.platform,
                id=str(conv.get("uuid") or conv.get("created_at")),
                title=conv.get("name") or "untitled",
                created_at=parse_iso(conv.get("created_at")),
                messages=messages,
            )
