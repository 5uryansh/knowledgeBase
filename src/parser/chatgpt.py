import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import hashlib


class ChatGPTChatParser:

    def __init__(self, input_file, output_dir):
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _extract_text(self, message):
        if message is None:
            return ""

        content = message.get("content", {})
        if content.get("content_type") != "text":
            return ""

        parts = content.get("parts", [])
        return "\n\n".join(
            str(part)
            for part in parts
            if str(part).strip()
        ).strip()

    def parse(self):
        with open(self.input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        grouped_by_date = defaultdict(list)

        sorted_data = sorted(data, key=lambda x: x.get("create_time", 0))

        for conversation in sorted_data:
            create_time = conversation.get("create_time")
            date_part = "unknown-date"
            if create_time:
                try:
                    dt = datetime.fromtimestamp(create_time)
                    date_part = dt.strftime("%Y-%m-%d")
                except Exception:
                    pass

            grouped_by_date[date_part].append(conversation)

        for date_part, conversations in grouped_by_date.items():
            markdown_parts = []
            combined_day_content = ""

            markdown_parts.append("---")
            markdown_parts.append("platform: chatgpt")
            markdown_parts.append(f"date: {date_part}")
            markdown_parts.append("---\n")

            markdown_parts.append(f"# ChatGPT Export - {date_part}\n")

            for conversation in conversations:
                title = conversation.get("title", "untitled")
                markdown_parts.append("\n---\n")
                markdown_parts.append(f"# {title}\n")
                mapping = conversation.get("mapping", {})

                messages = []
                for node in mapping.values():
                    message = node.get("message")

                    if message is None:
                        continue

                    if message.get("create_time") is None:
                        continue

                    messages.append(message)

                messages = sorted(messages, key=lambda x: x.get("create_time", 0))

                for message in messages:
                    role = (message.get("author", {}).get("role", ""))

                    text = self._extract_text(message)

                    if not text:
                        continue

                    timestamp = message.get("create_time")

                    try:
                        timestamp = datetime.fromtimestamp(
                            timestamp
                        ).isoformat()
                    except Exception:
                        timestamp = ""

                    if role == "user":
                        markdown_parts.append("\n## Prompt\n")
                    elif role == "assistant":
                        markdown_parts.append("\n## Response\n")
                    else:
                        continue

                    markdown_parts.append(text)

                    markdown_parts.append("\n### Timestamp\n")
                    markdown_parts.append(timestamp)

                    combined_day_content += text

            stable_hash = hashlib.sha256(
                combined_day_content.encode("utf-8")
            ).hexdigest()[:6]

            markdown_parts.insert(3, f"hash: {stable_hash}")

            final_markdown = "\n".join(markdown_parts)

            filename = (
                f"chatgpt_{date_part}_{stable_hash}.md"
            )

            output_path = (
                self.output_dir / filename
            )

            with open(output_path, "w", encoding="utf-8") as md_file:
                md_file.write(final_markdown)

        print(f"Generated markdown files: {self.output_dir}")