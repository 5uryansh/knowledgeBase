"""Query layer shared by the REST API, the MCP tools and the CLI."""
from __future__ import annotations

import re
import threading
from collections import Counter, defaultdict
from pathlib import Path

from knowledgebase.retrieval.index import KnowledgeIndex
from knowledgebase.retrieval.retriever import MODES
from knowledgebase.text import WIKILINK, strip_wikilinks

NODE_FIELDS = ("id", "label", "kind", "type", "importance", "community", "conversation_count", "mentions",
               "degree", "pagerank", "betweenness", "first_seen", "last_seen")
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
_ROLE = re.compile(r"\*\*(You|Assistant):\*\*\s*")


class NotFound(LookupError):
    pass


class KnowledgeService:
    def __init__(self, index_dir: Path, embedder: str = "auto", embed_model: str | None = None):
        self.index_dir = Path(index_dir)
        self.embedder_kind = embedder
        self.embed_model = embed_model
        self._index: KnowledgeIndex | None = None
        self._summaries: dict | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ lifecycle
    @property
    def index(self) -> KnowledgeIndex:
        if self._index is None:
            with self._lock:
                if self._index is None:
                    self._index = KnowledgeIndex.load(self.index_dir, embedder=self.embedder_kind,
                                                      embed_model=self.embed_model)
        return self._index

    @property
    def ready(self) -> bool:
        return (self.index_dir / "manifest.json").exists()

    def reload(self) -> None:
        with self._lock:
            self._index = None
            self._summaries = None

    @property
    def summaries(self) -> dict:
        if self._summaries is None:
            from knowledgebase.summaries import load_summaries

            self._summaries = load_summaries(self.index_dir)
        return self._summaries

    @property
    def graph(self):
        return self.index.graph

    # ------------------------------------------------------------------ helpers
    def node_summary(self, node_id: str) -> dict:
        node = self.graph.nodes[node_id]
        return {k: node.get(k) for k in NODE_FIELDS}

    def resolve(self, concept: str) -> str:
        """Accept a node id or any label/alias the user might type."""
        if concept in self.graph.nodes:
            return concept
        lowered = concept.strip().lower()
        if lowered in self.graph.nodes:
            return lowered
        normalized = re.sub(r"\s+", "_", lowered)
        if normalized in self.graph.nodes:
            return normalized
        hits = self.graph.search_nodes(concept, limit=1)
        if hits:
            return hits[0]["id"]
        raise NotFound(f"No concept matching {concept!r}")

    def _snippet(self, text: str, node_id: str | None = None, width: int = 220) -> str:
        if node_id:
            for sentence in _SENTENCE.split(text):
                if any(m.group(1).strip().lower() == node_id for m in WIKILINK.finditer(sentence)):
                    s = _ROLE.sub("", strip_wikilinks(sentence)).strip()
                    return s if len(s) <= width else s[:width].rsplit(" ", 1)[0] + " …"
        plain = _ROLE.sub("", strip_wikilinks(text)).strip()
        return plain if len(plain) <= width else plain[:width].rsplit(" ", 1)[0] + " …"

    # ------------------------------------------------------------------ stats / overview
    def stats(self) -> dict:
        idx = self.index
        return {
            **idx.manifest,
            "nodes": len(idx.graph.nodes),
            "edges": len(idx.graph.edges),
            "relations": len(idx.graph.relations),
            "communities": len(idx.graph.communities),
            "chunks": len(idx.chunks),
            "dense": idx.embedder is not None and idx.vectors is not None,
        }

    def overview(self, limit: int = 60, community: int | None = None, max_edges_per_node: int = 6) -> dict:
        g = self.graph
        if community is not None:
            members = [n for n in g.nodes.values() if n.get("community") == community]
            nodes = sorted(members, key=lambda n: -n.get("importance", 0))[:limit]
        else:
            nodes = g.top_nodes(limit)
        ids = {n["id"] for n in nodes}
        edges = []
        per_node: Counter = Counter()
        for e in sorted(g.edges_between(ids), key=lambda e: -e["strength"]):
            if per_node[e["source"]] >= max_edges_per_node and per_node[e["target"]] >= max_edges_per_node:
                continue
            per_node[e["source"]] += 1
            per_node[e["target"]] += 1
            edges.append(self._edge_view(e))
        if community is None and len(nodes) > 10:
            # Places with no route to anything else on the overview float off on their own and
            # just shrink the map; they are still one click away via search or a neighbour.
            connected = {e["source"] for e in edges} | {e["target"] for e in edges}
            nodes = [n for n in nodes if n["id"] in connected]
        return {
            "nodes": [self.node_summary(n["id"]) for n in nodes],
            "edges": edges,
            "communities": g.communities[:40],
            "meta": g.meta,
            "total_nodes": len(g.nodes),
        }

    @staticmethod
    def _edge_view(e: dict) -> dict:
        return {k: e[k] for k in ("source", "target", "strength", "npmi", "conversations", "weight") if k in e}

    # ------------------------------------------------------------------ node detail
    def mentions(self, node_id: str, limit: int = 20) -> list[dict]:
        idx = self.index
        by_conv: dict[str, list[int]] = defaultdict(list)
        for i in idx.by_concept.get(node_id, []):
            by_conv[idx.chunks[i].conversation_id].append(i)
        out = []
        for conv_id, chunk_ids in by_conv.items():
            first = idx.chunks[chunk_ids[0]]
            out.append({
                "conversation_id": conv_id,
                "title": first.title,
                "date": first.date,
                "platform": first.platform,
                "source": first.source,
                "chunk_id": first.id,
                "count": len(chunk_ids),
                "snippet": self._snippet(first.text, node_id),
            })
        out.sort(key=lambda m: (m["date"] or "", m["count"]), reverse=True)
        return out[:limit]

    def timeline(self, node_id: str) -> list[dict]:
        idx = self.index
        months: Counter = Counter()
        seen = set()
        for i in idx.by_concept.get(node_id, []):
            c = idx.chunks[i]
            if c.conversation_id in seen or not c.date or c.date == "unknown-date":
                continue
            seen.add(c.conversation_id)
            months[c.date[:7]] += 1
        return [{"month": m, "conversations": n} for m, n in sorted(months.items())]

    def worth_connecting(self, node_id: str, limit: int = 6, min_specificity: float = 0.45) -> list[dict]:
        """Concepts never mentioned together with ``node_id`` that share many strong neighbours with it
        (Adamic–Adar over the 2-hop neighbourhood) — ideas you may not have connected yet."""
        import math

        g = self.graph
        adj = g.adjacency
        direct = {o for o, _ in adj.get(node_id, [])}
        scores: dict[str, float] = defaultdict(float)
        via: dict[str, list[str]] = defaultdict(list)
        for mid, e1 in adj.get(node_id, [])[:30]:
            if e1.get("strength", 0) <= 0:
                continue
            deg = len(adj.get(mid, []))
            weight = 1.0 / math.log(deg + 1.5)
            for other, e2 in adj.get(mid, [])[:20]:
                if other == node_id or other in direct or e2.get("strength", 0) <= 0:
                    continue
                node = g.nodes.get(other)
                if not node or node.get("specificity", 1.0) < min_specificity or node.get("conversation_count", 0) < 2:
                    continue
                scores[other] += weight
                via[other].append(mid)
        ranked = sorted((o for o in scores if len(via[o]) >= 2), key=lambda o: -scores[o])[:limit]
        return [{**self.node_summary(o), "score": round(scores[o], 4),
                 "via": [{"id": m, "label": g.nodes[m]["label"]} for m in via[o][:3]]} for o in ranked]

    def node_detail(self, concept: str, limit: int = 25) -> dict:
        node_id = self.resolve(concept)
        g = self.graph
        neighbors = g.neighbors(node_id, limit=limit, min_strength=1e-9)
        ids = {node_id} | {o for o, _ in neighbors}
        rel_types: dict[str, set[str]] = defaultdict(set)
        relations = []
        for r in g.relations_by_node.get(node_id, []):
            other = r["target"] if r["source"] == node_id else r["source"]
            rel_types[other].add(r["relation"])
            relations.append({**r, "source_label": g.nodes[r["source"]]["label"],
                              "target_label": g.nodes[r["target"]]["label"]})
        neighbor_views = []
        for other, edge in neighbors:
            neighbor_views.append({
                **self.node_summary(other),
                "edge": self._edge_view(edge),
                "relations": sorted(rel_types.get(other, ())),
            })
        community = next((c for c in g.communities if c["id"] == g.nodes[node_id].get("community")), None)
        return {
            "node": {**g.nodes[node_id]},
            "community": community,
            "neighbors": neighbor_views,
            "edges": [self._edge_view(e) for e in g.edges_between(ids)],
            "relations": relations[:50],
            "mentions": self.mentions(node_id),
            "timeline": self.timeline(node_id),
            "worth_connecting": self.worth_connecting(node_id),
            "summary": self.summaries.get(node_id),
        }

    # ------------------------------------------------------------------ paths & discovery
    def connection(self, source: str, target: str) -> dict:
        a, b = self.resolve(source), self.resolve(target)
        path = self.graph.shortest_path(a, b)
        hops = []
        idx = self.index
        for x, y in zip(path, path[1:]):
            shared = sorted(set(idx.by_concept.get(x, [])) & set(idx.by_concept.get(y, [])))
            evidence = None
            if shared:
                c = idx.chunks[shared[-1]]
                evidence = {"chunk_id": c.id, "title": c.title, "date": c.date, "snippet": self._snippet(c.text, y)}
            edge = next((e for o, e in self.graph.adjacency.get(x, []) if o == y), None)
            hops.append({"source": x, "target": y, "strength": edge["strength"] if edge else None,
                         "evidence": evidence})
        return {"source": a, "target": b, "path": [self.node_summary(n) for n in path], "hops": hops}

    def discoveries(self) -> dict:
        g = self.graph
        label = lambda n: g.nodes[n]["label"] if n in g.nodes else n  # noqa: E731
        d = g.discoveries or {}
        comm_label = {c["id"]: c["label"] for c in g.communities}
        return {
            "bridges": [{**b, "label": label(b["node"]),
                         "community_labels": [comm_label.get(c, "") for c in b.get("communities", [])]}
                        for b in d.get("bridges", []) if b["node"] in g.nodes],
            "unexpected_links": [{**u, "source_label": label(u["source"]), "target_label": label(u["target"])}
                                 for u in d.get("unexpected_links", [])],
            "hidden_connections": [{**h, "source_label": label(h["source"]), "target_label": label(h["target"]),
                                    "via_labels": [label(v) for v in h.get("via", [])]}
                                   for h in d.get("hidden_connections", [])],
            "forgotten": [{**f, "label": label(f["node"])} for f in d.get("forgotten", []) if f["node"] in g.nodes],
        }

    # ------------------------------------------------------------------ search
    def search(self, query: str, k: int = 10, mode: str = "graph_hybrid", max_chars: int = 700) -> dict:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        idx = self.index
        results = idx.search(query, k=k, mode=mode)
        matched = idx.graph_retriever.matcher.match(query)
        concepts = [self.node_summary(n) for n in sorted(matched, key=lambda n: -matched[n]) if n in self.graph.nodes]
        window = idx.retriever.last_time_window
        return {
            "query": query,
            "mode": mode,
            "time_window": window.label if window else None,
            "results": [r.to_dict(max_chars=max_chars) for r in results],
            "concepts": concepts[:10],
        }

    def search_nodes(self, q: str, limit: int = 10) -> list[dict]:
        return [self.node_summary(n["id"]) for n in self.graph.search_nodes(q, limit=limit)]

    def activity(self, when: str = "last month", limit: int = 15, today=None) -> dict:
        """What you were working on in a period: conversations, recurring concepts and new ones."""
        from knowledgebase.retrieval.timefilter import parse_time_window

        window = parse_time_window(when, today) or parse_time_window(f"in {when}", today)
        if window is None:
            raise ValueError(f"Couldn't understand the period {when!r}; try 'last month', 'March 2026' or 'last spring'.")
        idx, g = self.index, self.graph
        hub = idx.graph_retriever.hub
        concept_pos = idx.graph_retriever.concept_idx
        convs, concept_convs = [], Counter()
        for conv_id, positions in idx.by_conversation.items():
            first = idx.chunks[positions[0]]
            if not window.matches(first.date):
                continue
            counts = Counter(l for i in positions for l in idx.chunks[i].links
                             if l in g.nodes and not hub[concept_pos[l]] and g.nodes[l].get("specificity", 1) >= 0.45)
            concept_convs.update(counts.keys())
            convs.append({"conversation_id": conv_id, "title": first.title, "date": first.date,
                          "platform": first.platform, "turns": len(positions),
                          "concepts": [g.nodes[c]["label"] for c, _ in counts.most_common(5)]})
        convs.sort(key=lambda c: c["date"], reverse=True)
        top = [{**self.node_summary(n), "conversations_in_period": k} for n, k in concept_convs.most_common(20)]
        new = [t for t in top if t.get("first_seen") and window.matches(t["first_seen"])]
        return {"period": window.label, "conversation_count": len(convs), "conversations": convs[:limit],
                "concepts": top, "new_concepts": new[:10]}

    def conversation(self, conversation_id: str, around: str | None = None, window: int = 3) -> dict:
        idx = self.index
        positions = idx.by_conversation.get(conversation_id)
        if not positions:
            raise NotFound(f"No conversation {conversation_id!r}")
        if around and around in idx.chunk_pos:
            p = idx.chunk_pos[around]
            local = positions.index(p) if p in positions else 0
            positions = positions[max(0, local - window): local + window + 1]
        chunks = [idx.chunks[i] for i in positions]
        first = chunks[0]
        return {
            "conversation_id": conversation_id,
            "title": first.title,
            "date": first.date,
            "platform": first.platform,
            "source": first.source,
            "chunks": [{"chunk_id": c.id, "turn": c.turn, "text": c.text, "concepts": c.links} for c in chunks],
        }
