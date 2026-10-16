"""LLM-written concept summaries: "what do I know about X?", from your own conversations.

    export KB_LLM_BASE_URL=http://localhost:11434/v1 KB_LLM_MODEL=qwen2.5:7b    # or GEMINI_API_KEY=...
    kb summarize --top 50

Summaries are stored in ``indexes/summaries.json`` and survive rebuilds. Each one
is keyed by a hash of its evidence, so re-running only pays for concepts whose
conversations changed. They show up in the atlas, the Obsidian notes and the
``explore_concept`` MCP tool.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

from knowledgebase.llm import LLM
from knowledgebase.text import WIKILINK, strip_wikilinks

PROMPT = """You are writing a short entry in someone's personal knowledge base.
Below are excerpts from their own past conversations with AI assistants that mention "{concept}".
Write what THEY learned, built, decided or struggled with regarding {concept}:
- 3 to 5 bullet points, one sentence each, concrete (names, numbers, decisions, fixes)
- if their understanding or choices changed over time, say how
- end every bullet with its source in square brackets: [conversation title, date]
- use only the excerpts; if they say little, write fewer bullets; never invent facts
Concepts closely connected to {concept} in their graph: {neighbors}

EXCERPTS:
{excerpts}

Answer with the bullet points only."""

_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
_ROLE = re.compile(r"\*\*(You|Assistant):\*\*\s*")


def load_summaries(index_dir: Path) -> dict[str, dict]:
    path = Path(index_dir) / "summaries.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


class Summarizer:
    def __init__(self, index, llm: LLM, index_dir: Path, max_passages: int = 8, passage_chars: int = 600):
        self.index, self.llm = index, llm
        self.path = Path(index_dir) / "summaries.json"
        self.max_passages, self.passage_chars = max_passages, passage_chars
        self.summaries = load_summaries(index_dir)

    def evidence(self, node_id: str) -> list[dict]:
        """One passage per conversation (newest first): the sentences around the concept's mentions."""
        idx = self.index
        best: dict[str, tuple[int, int]] = {}  # conversation -> (mentions in chunk, chunk index)
        for i in idx.by_concept.get(node_id, []):
            c = idx.chunks[i]
            n = sum(1 for m in WIKILINK.finditer(c.text) if m.group(1).strip().lower() == node_id)
            if c.conversation_id not in best or n > best[c.conversation_id][0]:
                best[c.conversation_id] = (n, i)
        chunks = sorted((idx.chunks[i] for _, i in best.values()), key=lambda c: c.date or "", reverse=True)
        out = []
        for c in chunks[: self.max_passages]:
            sentences = [s for s in _SENTENCE.split(c.text) if s.strip()]
            hits = [k for k, s in enumerate(sentences)
                    if any(m.group(1).strip().lower() == node_id for m in WIKILINK.finditer(s))]
            keep = sorted({j for k in hits for j in (k - 1, k, k + 1) if 0 <= j < len(sentences)}) or range(len(sentences))
            text = _ROLE.sub("", strip_wikilinks(" ".join(sentences[j] for j in keep))).strip()
            out.append({"title": c.title, "date": c.date, "conversation_id": c.conversation_id,
                        "text": text[: self.passage_chars]})
        return out

    def summarize(self, node_ids: list[str], force: bool = False, delay: float = 0.0, progress=None) -> int:
        graph = self.index.graph
        done = 0
        for node_id in node_ids:
            node = graph.nodes.get(node_id)
            if not node:
                continue
            evidence = self.evidence(node_id)
            if not evidence:
                continue
            key = hashlib.sha1(json.dumps([e["text"] for e in evidence]).encode()).hexdigest()[:12]
            cached = self.summaries.get(node_id)
            if cached and cached.get("evidence_key") == key and not force:
                continue
            neighbors = ", ".join(graph.nodes[o]["label"] for o, _ in graph.neighbors(node_id, limit=8))
            excerpts = "\n\n".join(f"[{e['title']}, {e['date']}] {e['text']}" for e in evidence)
            text = self.llm.complete(PROMPT.format(concept=node["label"], neighbors=neighbors or "none",
                                                   excerpts=excerpts)).strip()
            if not text:
                continue
            self.summaries[node_id] = {
                "summary": text,
                "model": self.llm.name,
                "evidence_key": key,
                "sources": [{k: e[k] for k in ("title", "date", "conversation_id")} for e in evidence],
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            self.save()
            done += 1
            if progress:
                progress(node["label"])
            if delay:
                time.sleep(delay)
        return done

    def save(self) -> None:
        self.path.write_text(json.dumps(self.summaries, ensure_ascii=False, indent=1), encoding="utf-8")
