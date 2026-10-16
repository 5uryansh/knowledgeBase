"""End-to-end build: exports -> conversations -> concepts -> vault -> graph -> index."""
from __future__ import annotations

import json
import logging
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from tqdm import tqdm

from knowledgebase import __version__
from knowledgebase.config import Settings
from knowledgebase.enrichment.concepts import Concept, ConceptExtractor
from knowledgebase.enrichment.linker import Linker
from knowledgebase.enrichment.obsidian_style import ObsidianGraphStyle
from knowledgebase.enrichment.vault import VaultWriter, conversation_relpath
from knowledgebase.enrichment.vocabulary import ConceptVocabulary
from knowledgebase.graph.analytics import compute_analytics
from knowledgebase.graph.cooccurrence import CooccurrenceBuilder
from knowledgebase.graph.model import KnowledgeGraph
from knowledgebase.graph.relations import RelationExtractor
from knowledgebase.memory_guard import check_memory_or_exit
from knowledgebase.parsers import PARSERS
from knowledgebase.parsers.base import Conversation
from knowledgebase.retrieval.chunk import Chunk
from knowledgebase.retrieval.chunker import Chunker
from knowledgebase.retrieval.embedders import Embedder, build_embedder
from knowledgebase.retrieval.index import KnowledgeIndex
from knowledgebase.text import WIKILINK

log = logging.getLogger(__name__)


@dataclass
class BuildReport:
    conversations: int
    chunks: int
    nodes: int
    edges: int
    relations: int
    communities: int
    embedder: str | None
    entity_extractor: str
    seconds: float

    def __str__(self) -> str:
        return (f"{self.conversations} conversations → {self.chunks} chunks · {self.nodes} concepts · "
                f"{self.edges} edges · {self.relations} relations · {self.communities} communities "
                f"[{self.entity_extractor}, {self.embedder or 'no dense'}] in {self.seconds:.1f}s")


def parse_exports(settings: Settings) -> list[Conversation]:
    conversations: dict[str, Conversation] = {}
    for platform, rel in settings.exports.items():
        parser_cls = PARSERS.get(platform)
        path = Path(rel).expanduser()
        if parser_cls is None or not path.exists():
            log.info("Skipping %s: %s not found", platform, path)
            continue
        log.info("Parsing %s export: %s", platform, path)
        for conv in parser_cls(path).parse():
            conversations[conv.uid] = conv  # re-imports overwrite instead of duplicating
    return sorted(conversations.values(), key=lambda c: (c.created_at is None, c.created_at or 0, c.uid))


class ConceptCache:
    """Concept extraction is the slow step; cache it per (conversation, content hash, extractor)."""

    def __init__(self, path: Path, signature: str):
        self.path, self.signature = path, signature
        self.data: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("sig") == signature:
                    self.data[row["uid"]] = row

    def get(self, conv: Conversation) -> list[Concept] | None:
        row = self.data.get(conv.uid)
        if row and row.get("hash") == conv.content_hash:
            return [Concept.from_dict(c) for c in row["concepts"]]
        return None

    def put(self, conv: Conversation, concepts: list[Concept]) -> None:
        self.data[conv.uid] = {"uid": conv.uid, "hash": conv.content_hash, "sig": self.signature,
                               "concepts": [c.to_dict() for c in concepts]}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            for row in self.data.values():
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def embed_with_cache(embedder: Embedder, chunks: list[Chunk], cache_dir: Path) -> np.ndarray:
    cache_file = cache_dir / f"emb_{embedder.name.replace('/', '_').replace(':', '_')}.npz"
    cached: dict[str, np.ndarray] = {}
    if cache_file.exists():
        with np.load(cache_file) as z:
            cached = dict(zip(z["keys"].tolist(), z["vecs"]))
    missing = [c for c in chunks if c.hash not in cached]
    if missing:
        log.info("Embedding %d chunks with %s (%d cached)", len(missing), embedder.name, len(chunks) - len(missing))
        vecs = embedder.embed([c.search_text for c in missing])
        for c, v in zip(missing, vecs):
            cached[c.hash] = v
    matrix = np.vstack([cached[c.hash] for c in chunks]).astype(np.float32) if chunks else \
        np.zeros((0, embedder.dim), dtype=np.float32)
    keep = {c.hash for c in chunks}
    cache_dir.mkdir(parents=True, exist_ok=True)
    keys = [k for k in cached if k in keep]
    np.savez(cache_file, keys=np.asarray(keys), vecs=np.vstack([cached[k] for k in keys]) if keys else matrix)
    return matrix


def build_graph(
    conversations: list[Conversation],
    vocabulary: ConceptVocabulary,
    linked_by_conv: dict[str, list[str]],
) -> KnowledgeGraph:
    cooc = CooccurrenceBuilder()
    relations = RelationExtractor()
    platforms: dict[str, Counter] = defaultdict(Counter)
    dates: dict[str, list[str]] = defaultdict(list)
    surfaces: dict[str, Counter] = defaultdict(Counter)
    for conv in conversations:
        linked = linked_by_conv[conv.uid]
        cooc.add_conversation(conv.uid, linked)
        seen: set[str] = set()
        for text in linked:
            relations.process(text, conv.uid)
            for m in WIKILINK.finditer(text):
                node = m.group(1).strip().lower()
                surfaces[node][m.group(2) or m.group(1)] += 1
                seen.add(node)
        for node in seen:
            platforms[node][conv.platform] += 1
            if conv.created_at:
                dates[node].append(conv.date)

    nodes: dict[str, dict] = {}
    for node_id, entry in vocabulary.entries.items():
        if cooc.node_mentions.get(node_id, 0) == 0:
            continue  # selected but never linked in text
        concept = entry.concept()
        d = sorted(dates.get(node_id, []))
        label = concept.display
        nodes[node_id] = {
            "id": node_id,
            "label": label,
            "kind": concept.kind,
            "type": concept.label,
            "aliases": [s for s, _ in surfaces[node_id].most_common(12) if s.lower() != label.lower()],
            "mentions": int(cooc.node_mentions.get(node_id, 0)),
            "conversation_count": len(cooc.node_conversations.get(node_id, ())),
            "platforms": dict(platforms[node_id]),
            "first_seen": d[0] if d else None,
            "last_seen": d[-1] if d else None,
        }
    edges = [e for e in cooc.export() if e["source"] in nodes and e["target"] in nodes]
    rels = [r for r in relations.export() if r["source"] in nodes and r["target"] in nodes]
    graph = KnowledgeGraph(nodes=nodes, edges=edges, relations=rels)
    return compute_analytics(graph)


def build(
    settings: Settings,
    conversations: Iterable[Conversation] | None = None,
    write_vault: bool = True,
    embedder: Embedder | None | str = "auto",
    extractor: ConceptExtractor | None = None,
    show_progress: bool = True,
) -> BuildReport:
    t0 = time.time()
    vault = Path(settings.vault_dir)
    index_dir = settings.index_dir
    index_dir.mkdir(parents=True, exist_ok=True)

    convs = list(conversations) if conversations is not None else parse_exports(settings)
    if not convs:
        raise SystemExit("No conversations found. Put your exports under data/ (see README) or set `exports:` in kb.yaml.")
    check_memory_or_exit(context="after parsing")

    extractor = extractor or ConceptExtractor(settings)
    cache = ConceptCache(index_dir / "cache" / "concepts.jsonl", extractor.signature)
    chunker = Chunker(max_chars=settings.max_chunk_chars)

    # Pass 1: propose concepts per conversation (cached — this is the slow step).
    concepts_by_conv: dict[str, list[Concept]] = {}
    for n, conv in enumerate(tqdm(convs, desc="Extracting concepts", disable=not show_progress), start=1):
        concepts = cache.get(conv)
        if concepts is None:
            concepts = extractor.extract(conv)
            cache.put(conv, concepts)
            if n % 50 == 0:
                cache.save()  # so a memory-guard stop doesn't lose finished work
        concepts_by_conv[conv.uid] = concepts
        check_memory_or_exit(context=f"while extracting {conv.title[:40]!r}")
    cache.save()

    # Pass 2: select a global vocabulary and link *every* conversation against it.
    vocabulary = ConceptVocabulary.build(concepts_by_conv)
    linker = Linker(vocabulary.concepts())
    linked_by_conv: dict[str, list[str]] = {}
    chunks: list[Chunk] = []
    for conv in tqdm(convs, desc="Linking", disable=not show_progress):
        linked = [linker.link(m.text) for m in conv.messages]
        linked_by_conv[conv.uid] = linked
        chunks.extend(chunker.chunk_conversation(conv, linked, source=conversation_relpath(conv)))

    graph = build_graph(convs, vocabulary, linked_by_conv)
    check_memory_or_exit(context="after building the graph")

    if isinstance(embedder, str):
        embedder = build_embedder(settings.embedder if embedder == "auto" else embedder, settings.embed_model)
    embeddings = embed_with_cache(embedder, chunks, index_dir / "cache") if embedder else None

    report = BuildReport(
        conversations=len(convs), chunks=len(chunks), nodes=len(graph.nodes), edges=len(graph.edges),
        relations=len(graph.relations), communities=len(graph.communities),
        embedder=embedder.name if embedder else None, entity_extractor=extractor.entity_extractor.name,
        seconds=0.0,
    )
    graph.meta = {
        "version": __version__,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "conversations": len(convs),
        "chunks": len(chunks),
        "platforms": dict(Counter(c.platform for c in convs)),
        "date_range": [min((c.date for c in convs if c.created_at), default=None),
                       max((c.date for c in convs if c.created_at), default=None)],
    }
    manifest = {
        **graph.meta,
        "embedder": embedder.name if embedder else None,
        "entity_extractor": extractor.entity_extractor.name,
        "concept_signature": extractor.signature,
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
    }
    KnowledgeIndex.save(index_dir, chunks, graph, embeddings, manifest)

    if write_vault:
        writer = VaultWriter(vault)
        writer.write_conversations([(c, linked_by_conv[c.uid]) for c in convs])
        mentions: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        for c in convs:
            rel = conversation_relpath(c)
            for node in {l for t in linked_by_conv[c.uid] for l in _links(t)}:
                mentions[node].append((rel, c.title, c.date))
        writer.write_concept_notes(graph, mentions)
        ObsidianGraphStyle(vault).apply()

    report.seconds = time.time() - t0
    return report


def _links(text: str) -> set[str]:
    from knowledgebase.graph.cooccurrence import links_in

    return set(links_in(text))
