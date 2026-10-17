"""The chat platforms knowledgeBase can read, as one enum.

Everything that depends on *which* AI an export came from (parser class, the files
to look for inside an export zip, how to tell one export from another, how users
get the export) hangs off :class:`Platform`, so adding a provider is one entry here
plus its parser.
"""
from __future__ import annotations

import json
import re
from enum import Enum
from pathlib import Path


class Platform(str, Enum):
    CHATGPT = "chatgpt"
    CLAUDE = "claude"
    CLAUDE_CODE = "claude-code"
    GEMINI = "gemini"

    @property
    def label(self) -> str:
        return _LABELS[self]

    @property
    def export_hint(self) -> str:
        return _HINTS[self]

    @property
    def parser(self):
        from knowledgebase.parsers import PARSERS

        return PARSERS[self.value]

    @property
    def reads_directory(self) -> bool:
        """Claude Code sessions are many .jsonl files; the parser takes their folder."""
        return self is Platform.CLAUDE_CODE

    def wants(self, member: str) -> bool:
        """Is this file (a path inside an export zip) one the parser needs?"""
        parts = Path(member).parts
        name = parts[-1] if parts else ""
        if self is Platform.CHATGPT:
            return bool(_CHATGPT_FILE.fullmatch(name))
        if self is Platform.CLAUDE:
            return name == "conversations.json"
        if self is Platform.GEMINI:
            # A full Takeout has a MyActivity.json per Google product; only Gemini's is a chat log.
            return name == "MyActivity.json" and any("Gemini" in part for part in parts[:-1])
        return name.endswith(".jsonl") and "subagents" not in parts

    @classmethod
    def parse(cls, value: str | Platform) -> Platform:
        if isinstance(value, Platform):
            return value
        key = str(value).strip().lower().replace("_", "-").replace(" ", "-")
        key = {"openai": "chatgpt", "claudecode": "claude-code", "bard": "gemini"}.get(key, key)
        try:
            return cls(key)
        except ValueError:
            raise ValueError(f"Unknown platform {value!r}. Choose one of: "
                             + ", ".join(p.value for p in cls)) from None


_CHATGPT_FILE = re.compile(r"conversations(?:-\d+)?\.json")

_LABELS = {
    Platform.CHATGPT: "ChatGPT",
    Platform.CLAUDE: "Claude",
    Platform.CLAUDE_CODE: "Claude Code",
    Platform.GEMINI: "Gemini",
}

_HINTS = {
    Platform.CHATGPT: "ChatGPT → Settings → Data controls → Export data. Upload the zip from the email.",
    Platform.CLAUDE: "claude.ai → Settings → Privacy → Export data. Upload the zip from the email.",
    Platform.CLAUDE_CODE: "Zip your ~/.claude/projects folder and upload it. "
                          "(On the machine running the server it is read automatically.)",
    Platform.GEMINI: "takeout.google.com → select only My Activity → Gemini Apps, "
                     "and set the format to JSON. Upload the Takeout zip.",
}


def sniff_json(path: Path) -> Platform | None:
    """Guess which platform a .json/.jsonl export file came from, from its structure."""
    try:
        if path.suffix == ".jsonl":
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        row = json.loads(line)
                        if isinstance(row, dict) and ("sessionId" in row or "message" in row):
                            return Platform.CLAUDE_CODE
                        return None
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    if not isinstance(data, list):
        return None
    for item in data[:20]:
        if not isinstance(item, dict):
            continue
        if "mapping" in item:
            return Platform.CHATGPT
        if "chat_messages" in item:
            return Platform.CLAUDE
        if "safeHtmlItem" in item or "Gemini Apps" in (item.get("header") or "") \
                or "Gemini Apps" in (item.get("products") or []):
            return Platform.GEMINI
    return None
