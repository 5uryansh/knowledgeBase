"""In-memory knowledge graph with query helpers used by the API, MCP tools and retriever."""
from __future__ import annotations

import heapq
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class KnowledgeGraph:
    nodes: dict[str, dict] = field(default_factory=dict)
    edges: list[dict] = field(default_factory=list)
    relations: list[dict] = field(default_factory=list)
    communities: list[dict] = field(default_factory=list)
    discoveries: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    _adj: dict[str, list[tuple[str, dict]]] | None = field(default=None, repr=False)
    _rel_by_node: dict[str, list[dict]] | None = field(default=None, repr=False)

    # ------------------------------------------------------------------ io
    def to_dict(self) -> dict:
        return {
            "meta": self.meta,
            "nodes": list(self.nodes.values()),
            "edges": self.edges,
            "relations": self.relations,
            "communities": self.communities,
            "discoveries": self.discoveries,
        }

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "KnowledgeGraph":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            nodes={n["id"]: n for n in data.get("nodes", [])},
            edges=data.get("edges", []),
            relations=data.get("relations", []),
            communities=data.get("communities", []),
            discoveries=data.get("discoveries", {}),
            meta=data.get("meta", {}),
        )

    # ------------------------------------------------------------------ indexes
    @property
    def adjacency(self) -> dict[str, list[tuple[str, dict]]]:
        if self._adj is None:
            adj: dict[str, list[tuple[str, dict]]] = defaultdict(list)
            for e in self.edges:
                adj[e["source"]].append((e["target"], e))
                adj[e["target"]].append((e["source"], e))
            for node in adj:
                adj[node].sort(key=lambda t: -t[1].get("strength", 0))
            self._adj = dict(adj)
        return self._adj

    @property
    def relations_by_node(self) -> dict[str, list[dict]]:
        if self._rel_by_node is None:
            by: dict[str, list[dict]] = defaultdict(list)
            for r in self.relations:
                by[r["source"]].append(r)
                by[r["target"]].append(r)
            self._rel_by_node = dict(by)
        return self._rel_by_node

    # ------------------------------------------------------------------ queries
    def neighbors(self, node_id: str, limit: int = 25, min_strength: float = 0.0) -> list[tuple[str, dict]]:
        out = []
        for other, edge in self.adjacency.get(node_id, []):
            if edge.get("strength", 0) < min_strength or other not in self.nodes:
                continue
            out.append((other, edge))
            if len(out) >= limit:
                break
        return out

    def edges_between(self, node_ids: set[str]) -> list[dict]:
        out, seen = [], set()
        for node in node_ids:
            for other, edge in self.adjacency.get(node, []):
                if other in node_ids and edge.get("strength", 0) > 0 and id(edge) not in seen:
                    seen.add(id(edge))
                    out.append(edge)
        return out

    def top_nodes(self, limit: int = 60, per_community_cap: float = 0.35) -> list[dict]:
        """Most important nodes, but diversified so one community can't take over the overview."""
        ranked = sorted(self.nodes.values(), key=lambda n: -n.get("importance", 0))
        cap = max(3, int(limit * per_community_cap))
        taken: dict[int, int] = defaultdict(int)
        chosen, rest = [], []
        for node in ranked:
            c = node.get("community", -1)
            if taken[c] < cap:
                chosen.append(node)
                taken[c] += 1
            else:
                rest.append(node)
            if len(chosen) >= limit:
                break
        if len(chosen) < limit:
            chosen.extend(rest[: limit - len(chosen)])
        return chosen

    def search_nodes(self, query: str, limit: int = 10) -> list[dict]:
        q = query.strip().lower()
        if not q:
            return []
        scored = []
        for node in self.nodes.values():
            names = [node["label"].lower(), node["id"].replace("_", " ")] + [a.lower() for a in node.get("aliases", [])]
            best = 0.0
            for name in names:
                if name == q:
                    best = max(best, 3.0)
                elif name.startswith(q):
                    best = max(best, 2.0)
                elif q in name:
                    best = max(best, 1.0)
            if best:
                scored.append((best + node.get("importance", 0), node))
        return [n for _, n in heapq.nlargest(limit, scored, key=lambda t: t[0])]

    def shortest_path(self, source: str, target: str, max_depth: int = 6) -> list[str]:
        """Strongest path: Dijkstra on cost = 1 / (strength + eps)."""
        if source not in self.nodes or target not in self.nodes:
            return []
        dist = {source: 0.0}
        prev: dict[str, str] = {}
        depth = {source: 0}
        heap = [(0.0, source)]
        while heap:
            d, node = heapq.heappop(heap)
            if node == target:
                break
            if d > dist.get(node, float("inf")) or depth[node] >= max_depth:
                continue
            for other, edge in self.adjacency.get(node, [])[:200]:
                s = edge.get("strength", 0)
                if s <= 0:
                    continue
                nd = d + 1.0 / (s + 1e-3)
                if nd < dist.get(other, float("inf")):
                    dist[other], prev[other], depth[other] = nd, node, depth[node] + 1
                    heapq.heappush(heap, (nd, other))
        if target not in dist:
            return []
        path = [target]
        while path[-1] != source:
            path.append(prev[path[-1]])
        return list(reversed(path))
