import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import hashlib
from src.parser.html_markdown_parser import HTMLToMarkdownParser

class GeminiChatParser:

    def __init__(self, input_file, output_dir):
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _html_to_markdown(self, html_text):
        parser = HTMLToMarkdownParser()
        parser.feed(html_text)
        return parser.get_markdown()

    def parse(self):

        with open(self.input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        grouped_by_date = defaultdict(list)

        sorted_data = sorted(
            data,
            key=lambda x: x.get("time", "")
        )

        for item in sorted_data:

            time = item.get("time", "")

            date_part = "unknown-date"

            if time:
                try:
                    dt = datetime.fromisoformat(
                        time.replace("Z", "+00:00")
                    )

                    date_part = dt.strftime("%Y-%m-%d")

                except Exception:
                    pass

            grouped_by_date[date_part].append(item)

        for date_part, items in grouped_by_date.items():

            markdown_parts = []

            combined_day_content = ""

            markdown_parts.append("---")
            markdown_parts.append("platform: gemini")
            markdown_parts.append(f"date: {date_part}")
            markdown_parts.append("---\n")

            markdown_parts.append(f"# Gemini Export - {date_part}\n")

            for item in items:

                title = item.get("title", "untitled")
                time = item.get("time", "")
                safe_html_items = item.get("safeHtmlItem", [])
                markdown_parts.append("\n---\n")
                markdown_parts.append(f"## Prompt\n")
                markdown_parts.append(f"{title}\n")
                markdown_parts.append(f"\n### Timestamp\n")
                markdown_parts.append(f"{time}\n")
                markdown_parts.append("\n## Response\n")
                response_parts = []

                for html_item in safe_html_items:
                    html_content = html_item.get("html", "")
                    markdown_text = self._html_to_markdown(html_content)
                    response_parts.append(markdown_text)
                response_text = "\n\n".join(response_parts)
                markdown_parts.append(response_text)
                combined_day_content += (title + response_text)
            stable_hash = hashlib.sha256(
                combined_day_content.encode("utf-8")).hexdigest()[:6]

            markdown_parts.insert(3, f"hash: {stable_hash}")
            final_markdown = "\n".join(markdown_parts)
            filename = (f"gemini_{date_part}_{stable_hash}.md")
            output_path = (self.output_dir / filename)

            with open(output_path, "w", encoding="utf-8") as md_file:
                md_file.write(final_markdown)

        print(f"Generated mardown files: {self.output_dir}")