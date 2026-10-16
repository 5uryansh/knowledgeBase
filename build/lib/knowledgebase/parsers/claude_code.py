"""Claude Code session parser (``~/.claude/projects/**/*.jsonl``)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from knowledgebase.parsers.base import Conversation, Message, parse_iso


class ClaudeCodeChatParser:
    platform = "claude-code"
    MAX_CHARS_PER_CONVERSATION = 700_000

    def __init__(self, input_dir: str | Path):
        self.input_dir = Path(input_dir)

    def _session_files(self) -> list[Path]:
        return sorted(f for f in self.input_dir.rglob("*.jsonl") if "subagents" not in f.parts)

    @staticmethod
    def _extract_text(entry: dict) -> str:
        content = (entry.get("message") or {}).get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
            return "\n\n".join(p for p in parts if p.strip()).strip()
        return ""

    def _load_session(self, path: Path) -> tuple[list[Message], str | None]:
        messages: list[Message] = []
        summary = None
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("type") == "summary" and entry.get("summary"):
                    summary = entry["summary"]
                # Skip meta entries and sub-agent "sidechains" (agent-internal turns, not your conversation).
                if entry.get("type") not in ("user", "assistant") or entry.get("isMeta") or entry.get("isSidechain"):
                    continue
                text = self._extract_text(entry)
                if not text or text.startswith(("<command-", "<local-command", "Caveat:")):
                    continue
                messages.append(Message(role=entry["type"], text=text, timestamp=entry.get("timestamp", "")))
        return messages, summary

    def _split(self, messages: list[Message]) -> list[list[Message]]:
        parts, current, size = [], [], 0
        for m in messages:
            if current and size + len(m.text) > self.MAX_CHARS_PER_CONVERSATION:
                parts.append(current)
                current, size = [], 0
            current.append(m)
            size += len(m.text)
        if current:
            parts.append(current)
        return parts

    def parse(self) -> Iterator[Conversation]:
        for session_file in self._session_files():
            messages, summary = self._load_session(session_file)
            if not messages:
                continue
            project = session_file.parent.name
            session_id = session_file.stem
            first_prompt = next((m.text for m in messages if m.role == "user"), "")
            base_title = summary or (first_prompt.splitlines()[0][:80] if first_prompt else session_id)
            parts = self._split(messages)
            for n, part in enumerate(parts, start=1):
                yield Conversation(
                    platform=self.platform,
                    id=f"{session_id}:{n}" if len(parts) > 1 else session_id,
                    title=f"{base_title} (part {n})" if len(parts) > 1 else base_title,
                    created_at=parse_iso(part[0].timestamp),
                    messages=part,
                    project=project,
                )
