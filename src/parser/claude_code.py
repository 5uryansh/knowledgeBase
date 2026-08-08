import json
from pathlib import Path
from datetime import datetime


class ClaudeCodeChatParser:

    MAX_CHARS_PER_FILE = 700_000

    def __init__(self, input_file, output_dir):
        self.input_dir = Path(input_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _find_session_files(self):
        return sorted(
            f for f in self.input_dir.rglob("*.jsonl")
            if "subagents" not in f.parts
        )

    def _extract_message_text(self, entry):
        content = entry.get("message", {}).get("content")

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts = [
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            return "\n\n".join(part for part in parts if part.strip()).strip()

        return ""

    def _load_session(self, session_file):
        messages = []
        with open(session_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                entry_type = entry.get("type")
                if entry_type not in ("user", "assistant"):
                    continue

                text = self._extract_message_text(entry)
                if not text:
                    continue

                messages.append({
                    "role": entry_type,
                    "text": text,
                    "timestamp": entry.get("timestamp", ""),
                })
        return messages

    def _split_messages(self, messages):
        parts = []
        current = []
        current_len = 0

        for message in messages:
            message_len = len(message["text"])
            if current and current_len + message_len > self.MAX_CHARS_PER_FILE:
                parts.append(current)
                current = []
                current_len = 0
            current.append(message)
            current_len += message_len

        if current:
            parts.append(current)

        return parts

    def parse(self):
        session_files = self._find_session_files()

        for session_file in session_files:
            messages = self._load_session(session_file)
            if not messages:
                continue

            first_timestamp = messages[0]["timestamp"]
            date_part = "unknown-date"
            if first_timestamp:
                try:
                    dt = datetime.fromisoformat(first_timestamp.replace("Z", "+00:00"))
                    date_part = dt.strftime("%Y-%m-%d")
                except Exception:
                    pass

            project_name = session_file.parent.name
            session_id = session_file.stem
            session_short_id = session_id.split("-")[0]

            message_parts = self._split_messages(messages)
            total_parts = len(message_parts)

            for part_number, part_messages in enumerate(message_parts, start=1):
                markdown_parts = []
                markdown_parts.append("---")
                markdown_parts.append("platform: claude-code")
                markdown_parts.append(f"date: {date_part}")
                markdown_parts.append(f"project: {project_name}")
                markdown_parts.append(f"session: {session_id}")
                if total_parts > 1:
                    markdown_parts.append(f"part: {part_number}/{total_parts}")
                markdown_parts.append("---\n")

                title = f"# {project_name} ({session_id})"
                if total_parts > 1:
                    title += f" — part {part_number}"
                markdown_parts.append(f"{title}\n")

                for message in part_messages:
                    if message["role"] == "user":
                        markdown_parts.append("\n## Prompt\n")
                    else:
                        markdown_parts.append("\n## Response\n")

                    markdown_parts.append(message["text"])
                    markdown_parts.append("\n### Timestamp\n")
                    markdown_parts.append(message["timestamp"])

                final_markdown = "\n".join(markdown_parts)
                if total_parts > 1:
                    filename = f"claudecode_{date_part}_{session_short_id}_{part_number}.md"
                else:
                    filename = f"claudecode_{date_part}_{session_short_id}.md"
                output_path = self.output_dir / filename

                with open(output_path, "w", encoding="utf-8") as md_file:
                    md_file.write(final_markdown)

        print(f"Generated markdown files: {self.output_dir}")
