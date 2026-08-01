from __future__ import annotations
import hashlib
import re
from pathlib import Path
from src.retrieval.chunk import Chunk

# Match [[entity]] and [[entity|alias]]
WIKILINK_PATTERN = re.compile(r"\[\[([^|\]]+)(?:\|[^\]]+)?\]\]")


class Chunker:
    def __init__(self, max_chars: int = 4000) -> None:
        self.max_chars = max_chars

    def chunk(self, markdown_file: Path) -> list[Chunk]:
        text = markdown_file.read_text(encoding="utf-8")
        sections = self._split_sections(text)
        chunks: list[Chunk] = []
        document_id = self._document_id(markdown_file)

        chunk_index = 0

        for section in sections:
            for piece in self._split_large_section(section):
                chunks.append(
                    Chunk(
                        id=f"{document_id}:{chunk_index}",
                        text=piece,
                        source=markdown_file,
                        links=self._extract_links(piece),
                        hash=self._hash(piece),
                    )
                )
                chunk_index += 1

        return chunks

    def _split_sections(self, text: str) -> list[str]:
        """
        Split on level-2 headings (## ...).
        The heading remains attached to its content.
        """
        matches = list(re.finditer(r"(?m)^##\s+", text))

        if not matches:
            return [text.strip()] if text.strip() else []

        sections: list[str] = []

        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

            section = text[start:end].strip()
            if section:
                sections.append(section)

        return sections

    def _split_large_section(self, section: str) -> list[str]:
        if len(section) <= self.max_chars:
            return [section]

        paragraphs = section.split("\n\n")

        chunks: list[str] = []
        current = ""

        for paragraph in paragraphs:
            if len(current) + len(paragraph) + 2 <= self.max_chars:
                current += ("\n\n" if current else "") + paragraph
            else:
                if current:
                    chunks.append(current)
                current = paragraph

        if current:
            chunks.append(current)

        return chunks

    def _extract_links(self, text: str) -> list[str]:
        return sorted(set(WIKILINK_PATTERN.findall(text)))

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    @staticmethod
    def _document_id(path: Path) -> str:
        return hashlib.sha1(str(path).encode()).hexdigest()[:12]