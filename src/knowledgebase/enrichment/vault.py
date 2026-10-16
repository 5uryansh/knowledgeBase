"""Writes the human-readable Obsidian vault.

    vault/
      conversations/<platform>/<date>_<slug>_<id>.md   one file per conversation
      entities/<node>.md                                one note per entity
      topics/<node>.md                                  one note per topic

Concept notes are regenerated on every build, but only between the
``<!-- kb:generated -->`` markers — anything you write outside them is kept.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from knowledgebase.graph.model import KnowledgeGraph
from knowledgebase.parsers.base import Conversation
from knowledgebase.text import WIKILINK, sanitize_filename, slugify

START = "<!-- kb:generated:start -->"
END = "<!-- kb:generated:end -->"
_GENERATED = re.compile(re.escape(START) + r".*?" + re.escape(END), re.DOTALL)


def conversation_relpath(conversation: Conversation) -> str:
    name = f"{conversation.date}_{slugify(conversation.title, 40)}_{conversation.uid[:6]}.md"
    return f"conversations/{conversation.platform}/{name}"


def notable(node: dict | None, min_specificity: float = 0.5) -> bool:
    """Concepts worth a link and a note in the vault. The index keeps every link (they help retrieval),
    but linking generic words like "update" or "region" everywhere just clutters Obsidian."""
    if node is None:
        return False
    if node.get("kind") == "entity" and node.get("type") not in ("topic", ""):
        return node.get("specificity", 1.0) >= 0.3
    return node.get("specificity", 1.0) >= min_specificity or node.get("conversation_count", 0) >= 4


def prune_links(text: str, keep: set[str]) -> str:
    return WIKILINK.sub(lambda m: m.group(0) if m.group(1).strip().lower() in keep else (m.group(2) or m.group(1)), text)


def render_conversation(conversation: Conversation, linked_texts: list[str]) -> str:
    front = {
        "platform": conversation.platform,
        "conversation_id": conversation.id,
        "uid": conversation.uid,
        "title": conversation.title,
        "date": conversation.date,
        "hash": conversation.content_hash,
    }
    if conversation.project:
        front["project"] = conversation.project
    lines = ["---"] + [f"{k}: {json.dumps(v, ensure_ascii=False)}" for k, v in front.items()] + ["---", ""]
    lines.append(f"# {conversation.title}\n")
    for message, text in zip(conversation.messages, linked_texts):
        lines.append("## Prompt" if message.role == "user" else "## Response")
        lines.append("")
        lines.append(text)
        if message.timestamp:
            lines.append("")
            lines.append("### Timestamp")
            lines.append("")
            lines.append(message.timestamp)
        lines.append("")
    return "\n".join(lines)


class VaultWriter:
    def __init__(self, vault_dir: Path):
        self.vault_dir = Path(vault_dir)
        self.manifest_path = self.vault_dir / "indexes" / "generated_files.json"

    def write_conversations(self, items: list[tuple[Conversation, list[str]]], keep: set[str] | None = None) -> list[str]:
        """Write conversation files; remove files this tool generated earlier that no longer exist.
        ``keep``: node ids that stay linked in the markdown (None = keep all)."""
        written = []
        for conversation, linked in items:
            if keep is not None:
                linked = [prune_links(t, keep) for t in linked]
            rel = conversation_relpath(conversation)
            path = self.vault_dir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            content = render_conversation(conversation, linked)
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                path.write_text(content, encoding="utf-8")
            written.append(rel)
        previous = []
        if self.manifest_path.exists():
            previous = json.loads(self.manifest_path.read_text(encoding="utf-8")).get("conversations", [])
        for rel in set(previous) - set(written):
            stale = self.vault_dir / rel
            if stale.exists():
                stale.unlink()
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps({"conversations": sorted(written)}, indent=1), encoding="utf-8")
        return written

    def write_concept_notes(self, graph: KnowledgeGraph, mentions: dict[str, list[tuple[str, str, str]]],
                            max_related: int = 15, max_mentions: int = 200) -> int:
        """``mentions[node] = [(conversation relpath, title, date), ...]``"""
        count = 0
        for node_id, node in graph.nodes.items():
            if not notable(node):
                continue
            folder = "entities" if node.get("kind") == "entity" else "topics"
            path = self.vault_dir / folder / f"{sanitize_filename(node_id)}.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            block = [START]
            meta = [node.get("type", node.get("kind", ""))]
            meta.append(f"{node.get('conversation_count', 0)} conversations")
            if node.get("first_seen"):
                meta.append(f"{node['first_seen']} → {node.get('last_seen', '')}")
            block.append(f"*{' · '.join(m for m in meta if m)}*")
            related = [(o, e) for o, e in graph.neighbors(node_id, limit=max_related * 3, min_strength=1e-6)
                       if notable(graph.nodes.get(o))][:max_related]
            if related:
                block.append("\n## Related")
                for other, edge in related:
                    block.append(f"- [[{other}|{graph.nodes[other]['label']}]] · strength {edge['strength']:.2f}")
            rels = [r for r in graph.relations_by_node.get(node_id, [])
                    if r["source"] == node_id and notable(graph.nodes.get(r["target"]))]
            if rels:
                block.append("\n## Relations")
                for r in rels[:30]:
                    block.append(f"- {r['relation']} → [[{r['target']}]]")
            convs = mentions.get(node_id, [])
            if convs:
                block.append("\n## Mentioned In")
                for rel, title, date in sorted(convs, key=lambda t: t[2], reverse=True)[:max_mentions]:
                    block.append(f"- [[{rel[:-3]}|{title}]] ({date})")
            block.append(END)
            generated = "\n".join(block)
            if path.exists():
                existing = path.read_text(encoding="utf-8")
                content = _GENERATED.sub(lambda _m: generated, existing) if START in existing else \
                    existing.rstrip() + "\n\n" + generated + "\n"
            else:
                content = f"# {node['label']}\n\n{generated}\n"
            path.write_text(content, encoding="utf-8")
            count += 1
        return count
