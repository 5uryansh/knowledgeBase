import json
from pathlib import Path
from datetime import datetime
from html.parser import HTMLParser


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

        markdown_parts = []

        markdown_parts.append("# Gemini Conversation Export\n")

        sorted_data = sorted(
            data,
            key=lambda x: x.get("time", "")
        )

        for item in sorted_data:

            title = item.get("title", "untitled")
            time = item.get("time", "")
            safe_html_items = item.get("safeHtmlItem", [])

            markdown_parts.append("---\n")

            if time:
                markdown_parts.append(f"## {time}\n")
            else:
                markdown_parts.append("## Unknown Time\n")

            markdown_parts.append(f"### Prompt\n")
            markdown_parts.append(f"{title}\n")

            markdown_parts.append("### Response\n")

            for html_item in safe_html_items:

                html_content = html_item.get("html", "")
                markdown_text = self._html_to_markdown(html_content)

                markdown_parts.append(markdown_text)
                markdown_parts.append("\n")

        final_markdown = "\n".join(markdown_parts)
        final_markdown = final_markdown.replace("[['text', 'label']]", "")

        output_path = self.output_dir / "gemini_export.md"

        with open(output_path, "w", encoding="utf-8") as md_file:
            md_file.write(final_markdown)

        print(f"Generated: {output_path}")