"""Conversation-aware chunking.

A chunk is a *question/answer pair*: the user's prompt together with the reply.
Personal-recall queries ("where did I fix X?") usually mirror the user's own
wording, while the answer carries the details — keeping them together beats
chunking either side alone. Over-long pairs are split on paragraph boundaries,
and every piece keeps the prompt's first line as context.
"""
from __future__ import annotations

import hashlib

from knowledgebase.parsers.base import Conversation
from knowledgebase.retrieval.chunk import Chunk
from knowledgebase.text import WIKILINK


def _links(text: str) -> list[str]:
    return sorted({m.group(1).strip().lower() for m in WIKILINK.finditer(text)})


class Chunker:
    def __init__(self, max_chars: int = 2400) -> None:
        self.max_chars = max_chars

    def _split(self, text: str) -> list[str]:
        if len(text) <= self.max_chars:
            return [text]
        pieces, current = [], ""
        for paragraph in text.split("\n\n"):
            while len(paragraph) > self.max_chars:  # a single giant paragraph / code block
                head, paragraph = paragraph[: self.max_chars], paragraph[self.max_chars:]
                if current:
                    pieces.append(current)
                    current = ""
                pieces.append(head)
            if len(current) + len(paragraph) + 2 <= self.max_chars:
                current = f"{current}\n\n{paragraph}" if current else paragraph
            else:
                if current:
                    pieces.append(current)
                current = paragraph
        if current:
            pieces.append(current)
        return pieces

    def chunk_conversation(self, conversation: Conversation, linked_texts: list[str], source: str = "") -> list[Chunk]:
        """``linked_texts[i]`` is the link-enriched text of ``conversation.messages[i]``."""
        groups: list[tuple[int, list[str]]] = []
        for i, (message, text) in enumerate(zip(conversation.messages, linked_texts)):
            label = "**You:**" if message.role == "user" else "**Assistant:**"
            block = f"{label} {text}"
            if message.role == "user" or not groups:
                groups.append((i, [block]))
            else:
                groups[-1][1].append(block)

        chunks: list[Chunk] = []
        for turn, blocks in groups:
            body = "\n\n".join(blocks)
            pieces = self._split(body)
            context = blocks[0].split("\n", 1)[0][:200] if len(pieces) > 1 else ""
            for j, piece in enumerate(pieces):
                if j > 0 and context:
                    piece = f"{context} …\n\n{piece}"
                cid = f"{conversation.uid}:{turn}" + (f".{j}" if len(pieces) > 1 else "")
                chunks.append(Chunk(
                    id=cid,
                    conversation_id=conversation.uid,
                    title=conversation.title,
                    platform=conversation.platform,
                    date=conversation.date,
                    turn=turn,
                    text=piece,
                    links=_links(piece),
                    source=source,
                    hash=hashlib.sha1(piece.encode("utf-8")).hexdigest()[:16],
                ))
        return chunks
