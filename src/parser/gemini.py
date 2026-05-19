# parser/chat_parser.py

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from bs4 import BeautifulSoup


class GeminiParser:
    """
    Parser for Gemini exported conversation JSON files.
    """

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.raw_data: List[Dict[str, Any]] = []

    def load(self) -> None:
        """
        Load JSON export file.
        """
        with open(self.file_path, "r", encoding="utf-8") as f:
            self.raw_data = json.load(f)

    def _extract_html_text(self, html_content: str) -> str:
        """
        Convert Gemini safeHtmlItem HTML into plain text.
        """
        soup = BeautifulSoup(html_content, "html.parser")
        return soup.get_text(separator="\n").strip()

    def parse_conversations(self) -> List[Dict[str, Any]]:
        """
        Parse all Gemini conversations into structured format.
        """

        parsed_conversations = []

        for item in self.raw_data:

            conversation = {
                "source": "gemini",
                "header": item.get("header"),
                "title": item.get("title"),
                "time": item.get("time"),
                "products": item.get("products", []),
                "activity_controls": item.get("activityControls", []),
                "messages": []
            }

            safe_html_items = item.get("safeHtmlItem", [])

            for html_item in safe_html_items:

                html_content = html_item.get("html", "")

                parsed_message = {
                    "type": "html",
                    "raw_html": html_content,
                    "text": self._extract_html_text(html_content)
                }

                conversation["messages"].append(parsed_message)

            parsed_conversations.append(conversation)

        return parsed_conversations


if __name__ == "__main__":

    parser = GeminiParser("gemini_export.json")

    parser.load()

    conversations = parser.parse_conversations()

    print(json.dumps(conversations, indent=2, ensure_ascii=False))