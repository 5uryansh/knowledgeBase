import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import hashlib


class ClaudeChatParser:

    def __init__(self, input_file, output_dir):
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _extract_message_text(self, message):
        parts = []
        for content in message.get("content", []):
            content_type = content.get("type")
            if content_type == "text":
                text = content.get("text", "")
                if text.strip():
                    parts.append(text)
            elif content_type == "thinking":
                thinking = content.get("thinking", "")
                if thinking.strip():
                    parts.append(f"\n> Claude Thinking\n>\n> {thinking}\n")
        return "\n\n".join(parts).strip()

    def parse(self):
        with open(self.input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        grouped_by_date = defaultdict(list)
        sorted_data = sorted(data, key=lambda x: x.get("created_at", ""))

        for conversation in sorted_data:
            created_at = conversation.get("created_at", "")
            date_part = "unknown-date"
            if created_at:
                try:
                    dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    date_part = dt.strftime("%Y-%m-%d")
                except Exception:
                    pass
            grouped_by_date[date_part].append(conversation)

        for date_part, conversations in grouped_by_date.items():
            markdown_parts = []
            combined_day_content = ""
            markdown_parts.append("---")
            markdown_parts.append("platform: claude")
            markdown_parts.append(f"date: {date_part}")
            markdown_parts.append("---\n")
            markdown_parts.append(f"# Claude Export - {date_part}\n")

            for conversation in conversations:
                title = (conversation.get("name") or "untitled")
                markdown_parts.append("\n---\n")
                markdown_parts.append(f"# {title}\n")
                messages = conversation.get("chat_messages", [])
                sorted_messages = sorted(messages, key=lambda x: x.get("created_at", ""))
                for message in sorted_messages:
                    sender = message.get("sender", "")
                    timestamp = message.get("created_at", "")
                    text = self._extract_message_text(message)

                    if not text.strip():
                        continue

                    if sender == "human":
                        markdown_parts.append("\n## Prompt\n")
                    else:
                        markdown_parts.append("\n## Response\n")

                    markdown_parts.append(text)
                    markdown_parts.append("\n### Timestamp\n")
                    markdown_parts.append(timestamp)
                    combined_day_content += text

            stable_hash = hashlib.sha256(combined_day_content.encode("utf-8")).hexdigest()[:6]
            markdown_parts.insert(3, f"hash: {stable_hash}")

            final_markdown = "\n".join(markdown_parts)
            filename = (f"claude_{date_part}_{stable_hash}.md")
            output_path = (self.output_dir / filename)

            with open(output_path, "w", encoding="utf-8") as md_file:
                md_file.write(final_markdown)

        print(f"Generated markdown files: {self.output_dir}")