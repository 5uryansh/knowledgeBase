import json
from pathlib import Path
from datetime import datetime
from html.parser import HTMLParser
import hashlib


class HTMLToMarkdownParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in ["p", "div"]:
            self.parts.append("\n")
        elif tag in ["h1", "h2", "h3", "h4"]:
            self.parts.append("\n## ")
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag == "br":
            self.parts.append("\n")
        elif tag == "strong":
            self.parts.append("**")
        elif tag == "em":
            self.parts.append("*")
        elif tag == "code":
            self.parts.append("`")

    def handle_endtag(self, tag):
        if tag == "strong":
            self.parts.append("**")
        elif tag == "em":
            self.parts.append("*")
        elif tag == "code":
            self.parts.append("`")
        elif tag in ["p", "ul", "ol"]:
            self.parts.append("\n")

    def handle_data(self, data):
        cleaned = data.strip()
        if not cleaned:
            return
        if cleaned in ["text", "label"]:
            return
        self.parts.append(cleaned)

    def get_markdown(self):
        return "".join(self.parts).strip()


class GeminiChatParser:

    def __init__(self, input_file, output_dir):
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _html_to_markdown(self, html_text):
        parser = HTMLToMarkdownParser()
        parser.feed(html_text)
        return parser.get_markdown()

    def _sanitize_filename(self, text):
        return "".join(
            c if c.isalnum() or c in ["-", "_"] else "_"
            for c in text[:80]
        )

    def parse(self):

        with open(self.input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        sorted_data = sorted(
            data,
            key=lambda x: x.get("time", "")
        )

        for item in sorted_data:

            title = item.get("title", "untitled")
            time = item.get("time", "")
            source = item.get("header", "Gemini")
            safe_html_items = item.get("safeHtmlItem", [])

            markdown_parts = []

            conversation_text = []

            for html_item in safe_html_items:

                html_content = html_item.get("html", "")
                markdown_text = self._html_to_markdown(html_content)

                conversation_text.append(markdown_text)

            response_text = "\n\n".join(conversation_text)

            raw_hash_input = f"{title}_{time}_{response_text}"

            stable_hash = hashlib.sha256(
                raw_hash_input.encode("utf-8")
            ).hexdigest()[:6]

            markdown_parts.append("---")
            markdown_parts.append(f"platform: gemini")
            markdown_parts.append(f"source: {source}")

            if time:
                markdown_parts.append(f"timestamp: {time}")

            markdown_parts.append(f"hash: {stable_hash}")
            markdown_parts.append("---\n")

            markdown_parts.append(f"# {title}\n")

            markdown_parts.append("## Response\n")
            markdown_parts.append(response_text)

            final_markdown = "\n".join(markdown_parts)

            date_part = "unknown-date"

            if time:
                try:
                    dt = datetime.fromisoformat(
                        time.replace("Z", "+00:00")
                    )
                    date_part = dt.strftime("%Y-%m-%d")
                except Exception:
                    pass

            filename = (
                f"gemini_{date_part}_{stable_hash}.md"
            )

            output_path = self.output_dir / filename

            with open(output_path, "w", encoding="utf-8") as md_file:
                md_file.write(final_markdown)

            print(f"Generated: {output_path}")