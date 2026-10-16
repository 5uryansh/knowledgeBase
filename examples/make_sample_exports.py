"""Generate the sample chat exports in each platform's *native* export format.

    python examples/make_sample_exports.py

Writes examples/sample_exports/{chatgpt,claude,gemini,claude-code}/... — the same
files you would get from ChatGPT's data export, Claude's export, Google Takeout
and ~/.claude/projects, so `kb demo` exercises every real parser.
"""
from __future__ import annotations

import hashlib
import html
import json
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from sample_data import part1, part2, part3  # noqa: E402

OUT = HERE / "sample_exports"
CONVERSATIONS = part1.CONVERSATIONS + part2.CONVERSATIONS + part3.CONVERSATIONS
PROJECTS = {
    "esp32-weather": "-home-alex-projects-esp32-weather",
    "notes-app": "-home-alex-projects-notes-app",
    "knowledge-graph repo": "-home-alex-projects-knowledge-graph",
}


def _uuid(*parts: str) -> str:
    return str(uuid.UUID(hashlib.md5("|".join(parts).encode()).hexdigest()))


def _times(date: str, n: int) -> list[datetime]:
    start = datetime.fromisoformat(date).replace(hour=19, minute=12, tzinfo=timezone.utc)
    return [start + timedelta(minutes=3 * i) for i in range(n)]


def chatgpt(convs):
    out = []
    for date, title, turns in convs:
        times = _times(date, len(turns))
        mapping, parent = {}, None
        root = _uuid(title, "root")
        mapping[root] = {"id": root, "message": None, "parent": None, "children": []}
        parent = root
        for i, (text, ts) in enumerate(zip(turns, times)):
            node = _uuid(title, str(i))
            mapping[node] = {
                "id": node,
                "message": {
                    "id": node,
                    "author": {"role": "user" if i % 2 == 0 else "assistant"},
                    "create_time": ts.timestamp(),
                    "content": {"content_type": "text", "parts": [text]},
                },
                "parent": parent,
                "children": [],
            }
            mapping[parent]["children"].append(node)
            parent = node
        out.append({"title": title, "create_time": times[0].timestamp(), "update_time": times[-1].timestamp(),
                    "mapping": mapping, "current_node": parent, "conversation_id": _uuid(title)})
    return out


def claude(convs):
    out = []
    for date, title, turns in convs:
        times = _times(date, len(turns))
        out.append({
            "uuid": _uuid(title),
            "name": title,
            "created_at": times[0].isoformat().replace("+00:00", "Z"),
            "chat_messages": [
                {"uuid": _uuid(title, str(i)), "sender": "human" if i % 2 == 0 else "assistant",
                 "created_at": ts.isoformat().replace("+00:00", "Z"),
                 "content": [{"type": "text", "text": text}]}
                for i, (text, ts) in enumerate(zip(turns, times))
            ],
        })
    return out


def gemini(convs):
    items = []
    for date, _title, turns in convs:
        times = _times(date, len(turns))
        for i in range(0, len(turns) - 1, 2):
            paragraphs = "".join(f"<p>{html.escape(p)}</p>" for p in turns[i + 1].split("\n\n"))
            items.append({"header": "Gemini Apps", "title": f"Prompted {turns[i]}",
                          "time": times[i].isoformat().replace("+00:00", "Z"),
                          "products": ["Gemini Apps"], "safeHtmlItem": [{"html": paragraphs}]})
    return sorted(items, key=lambda x: x["time"], reverse=True)  # Takeout lists newest first


def claude_code(convs, root: Path):
    for date, title, turns in convs:
        project = next((v for k, v in PROJECTS.items() if title.startswith(k)), "-home-alex-projects-misc")
        session = _uuid(title)
        path = root / project / f"{session}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        times = _times(date, len(turns))
        lines = [{"type": "summary", "summary": title, "leafUuid": _uuid(title, "leaf")}]
        for i, (text, ts) in enumerate(zip(turns, times)):
            role = "user" if i % 2 == 0 else "assistant"
            content = text if role == "user" else [{"type": "text", "text": text}]
            lines.append({"type": role, "sessionId": session, "uuid": _uuid(title, str(i)),
                          "timestamp": ts.isoformat().replace("+00:00", "Z"),
                          "message": {"role": role, "content": content}})
        path.write_text("\n".join(json.dumps(l, ensure_ascii=False) for l in lines) + "\n", encoding="utf-8")


def main() -> None:
    by_platform: dict[str, list] = {"chatgpt": [], "claude": [], "gemini": [], "claude-code": []}
    for platform, date, title, turns in CONVERSATIONS:
        by_platform[platform].append((date, title, turns))
    (OUT / "chatgpt").mkdir(parents=True, exist_ok=True)
    (OUT / "claude").mkdir(parents=True, exist_ok=True)
    (OUT / "gemini").mkdir(parents=True, exist_ok=True)
    (OUT / "chatgpt" / "conversations.json").write_text(json.dumps(chatgpt(by_platform["chatgpt"]), indent=1), encoding="utf-8")
    (OUT / "claude" / "conversations.json").write_text(json.dumps(claude(by_platform["claude"]), indent=1), encoding="utf-8")
    (OUT / "gemini" / "MyActivity.json").write_text(json.dumps(gemini(by_platform["gemini"]), indent=1), encoding="utf-8")
    claude_code(by_platform["claude-code"], OUT / "claude-code" / "projects")
    counts = {k: len(v) for k, v in by_platform.items()}
    print(f"wrote {sum(counts.values())} conversations to {OUT}: {counts}")


if __name__ == "__main__":
    main()
