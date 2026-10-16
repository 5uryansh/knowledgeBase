"""ChatGPT export parser (``conversations.json`` from Settings → Data controls → Export)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from knowledgebase.parsers.base import Conversation, Message, parse_epoch


class ChatGPTChatParser:
    platform = "chatgpt"

    def __init__(self, input_file: str | Path):
        self.input_file = Path(input_file)

    @staticmethod
    def _extract_text(message: dict | None) -> str:
        if not message:
            return ""
        content = message.get("content") or {}
        if content.get("content_type") not in ("text", "multimodal_text"):
            return ""
        parts = content.get("parts") or []
        return "\n\n".join(str(p) for p in parts if isinstance(p, str) and p.strip()).strip()

    def _ordered_messages(self, conversation: dict) -> list[dict]:
        """Follow the active branch (current_node -> root) instead of mixing edited branches."""
        mapping = conversation.get("mapping") or {}
        node_id = conversation.get("current_node")
        chain: list[dict] = []
        seen: set[str] = set()
        while node_id and node_id in mapping and node_id not in seen:
            seen.add(node_id)
            node = mapping[node_id]
            if node.get("message"):
                chain.append(node["message"])
            node_id = node.get("parent")
        if chain:
            return list(reversed(chain))
        # Fallback for exports without current_node: chronological order.
        messages = [n["message"] for n in mapping.values() if n.get("message")]
        return sorted(messages, key=lambda m: m.get("create_time") or 0)

    def parse(self) -> Iterator[Conversation]:
        data = json.loads(self.input_file.read_text(encoding="utf-8"))
        for conv in sorted(data, key=lambda c: c.get("create_time") or 0):
            messages: list[Message] = []
            for message in self._ordered_messages(conv):
                role = (message.get("author") or {}).get("role", "")
                if role not in ("user", "assistant"):
                    continue
                text = self._extract_text(message)
                if not text:
                    continue
                ts = parse_epoch(message.get("create_time"))
                messages.append(Message(role=role, text=text, timestamp=ts.isoformat() if ts else ""))
            if not messages:
                continue
            yield Conversation(
                platform=self.platform,
                id=str(conv.get("conversation_id") or conv.get("id") or conv.get("create_time")),
                title=conv.get("title") or "untitled",
                created_at=parse_epoch(conv.get("create_time")),
                messages=messages,
            )
