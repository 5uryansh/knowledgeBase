"""Graph analytics: importance, communities, bridges and *discoveries*.

Discoveries are the "what did I not notice?" layer that makes the graph more than
a hairball:

* **bridges** — concepts that connect otherwise separate areas of your thinking
* **unexpected links** — strong associations between different communities
* **hidden connections** — concepts never discussed together that share many
  neighbours (candidates for an idea you haven't connected yet)
* **forgotten** — important concepts you haven't touched in a long time
"""
from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from datetime import date

import networkx as nx

from knowledgebase.graph.model import KnowledgeGraph
from knowledgebase.graph.specificity import specificity

log = logging.getLogger(__name__)


def _to_nx(graph: KnowledgeGraph) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(graph.nodes)
    for e in graph.edges:
        s = e.get("strength", 0)
        if s > 0 and e["source"] in graph.nodes and e["target"] in graph.nodes:
            G.add_edge(e["source"], e["target"], weight=s, distance=1.0 / s)
    return G


def _parse_date(value: str | None) -> date | None:
    try:
        return date.fromisoformat(value[:10]) if value else None
    except ValueError:
        return None


def compute_analytics(graph: KnowledgeGraph, seed: int = 42) -> KnowledgeGraph:
    G = _to_nx(graph)
    n = G.number_of_nodes()
    if n == 0:
        return graph

    pagerank = nx.pagerank(G, weight="weight") if G.number_of_edges() else {v: 1.0 / n for v in G}

    communities = []
    if G.number_of_edges():
        communities = nx.community.louvain_communities(G, weight="weight", seed=seed, resolution=1.0)
    community_of: dict[str, int] = {}
    for idx, members in enumerate(sorted(communities, key=len, reverse=True)):
        for m in members:
            community_of[m] = idx

    k = min(n, 300)
    betweenness = (
        nx.betweenness_centrality(G, k=k, weight="distance", seed=seed, normalized=True)
        if G.number_of_edges()
        else {}
    )

    max_pr = max(pagerank.values()) or 1.0
    for node_id, node in graph.nodes.items():
        pr = pagerank.get(node_id, 0.0)
        node["pagerank"] = round(pr, 6)
        node["community"] = community_of.get(node_id, -1)
        node["betweenness"] = round(betweenness.get(node_id, 0.0), 6)
        node["degree"] = G.degree(node_id) if node_id in G else 0
        # Centrality blends structural position with how often you actually came back to it;
        # importance (what the explorer shows first) also favours specific concepts over generic words.
        centrality = 0.6 * (pr / max_pr) + 0.4 * min(1.0, math.log1p(node.get("conversation_count", 0)) / math.log(50))
        spec = specificity(node.get("label", node_id), node.get("kind", "topic"), node.get("type", ""))
        node["specificity"] = spec
        node["centrality"] = round(centrality, 6)
        node["importance"] = round(centrality * (0.3 + 0.7 * spec), 6)

    # Community summaries
    members_by_c: dict[int, list[str]] = defaultdict(list)
    for node_id, c in community_of.items():
        members_by_c[c].append(node_id)
    graph.communities = []
    for c, members in sorted(members_by_c.items()):
        members.sort(key=lambda m: -graph.nodes[m]["importance"])
        top = members[:5]
        graph.communities.append({
            "id": c,
            "label": " & ".join(graph.nodes[m]["label"] for m in top[:2]),
            "size": len(members),
            "top_nodes": top,
        })

    graph.discoveries = discover(graph, G)
    graph._adj = None
    return graph


MIN_SPECIFICITY = 0.45


def discover(graph: KnowledgeGraph, G: nx.Graph, limit: int = 12) -> dict:
    nodes = graph.nodes
    specific = {n for n, d in nodes.items() if d.get("specificity", 1.0) >= MIN_SPECIFICITY}
    out: dict[str, list] = {"bridges": [], "unexpected_links": [], "hidden_connections": [], "forgotten": []}
    if G.number_of_edges() == 0:
        return out

    # Bridges: high betweenness + neighbours spread over several communities.
    for node_id in sorted(specific, key=lambda v: -nodes[v].get("betweenness", 0))[: limit * 4]:
        nbr_comms = Counter(nodes[o].get("community", -1) for o in G.neighbors(node_id))
        own = nodes[node_id].get("community", -1)
        others = [c for c, cnt in nbr_comms.most_common() if c != own and cnt >= 1]
        if len(others) >= 1 and nodes[node_id].get("conversation_count", 0) >= 2:
            out["bridges"].append({
                "node": node_id,
                "communities": [own] + others[:3],
                "score": nodes[node_id]["betweenness"],
            })
        if len(out["bridges"]) >= limit:
            break

    # Unexpected links: strong NPMI between different communities, seen in >=2 conversations.
    cross = [
        e for e in graph.edges
        if e.get("strength", 0) > 0
        and e["source"] in specific and e["target"] in specific
        and nodes[e["source"]].get("community") != nodes[e["target"]].get("community")
        and e.get("conversations", 0) >= 2
    ]
    spec = lambda n: nodes[n].get("specificity", 1.0)  # noqa: E731
    cross.sort(key=lambda e: -(e["npmi"] * math.log1p(e["conversations"]) * spec(e["source"]) * spec(e["target"])))
    out["unexpected_links"] = [
        {"source": e["source"], "target": e["target"], "npmi": e["npmi"], "conversations": e["conversations"]}
        for e in cross[:limit]
    ]

    # Hidden connections: not adjacent, many strong shared neighbours (Adamic–Adar), among important nodes.
    # Hidden connections: recurring concepts that were *never* mentioned in the same context.
    ever_together = {(e["source"], e["target"]) for e in graph.edges} | {(e["target"], e["source"]) for e in graph.edges}
    candidates = sorted((n for n in specific if nodes[n].get("conversation_count", 0) >= 2),
                        key=lambda v: -nodes[v].get("importance", 0))[:400]
    cand_set = set(candidates)
    pairs = set()
    for node_id in candidates:
        nbrs = sorted(G[node_id].items(), key=lambda t: -t[1]["weight"])[:15]
        for mid, _ in nbrs:
            for other, _w in sorted(G[mid].items(), key=lambda t: -t[1]["weight"])[:15]:
                if other != node_id and other in cand_set and (node_id, other) not in ever_together:
                    pairs.add((min(node_id, other), max(node_id, other)))
    scored = []
    for a, b, score in nx.adamic_adar_index(G, list(pairs)):
        shared = sorted(set(G[a]) & set(G[b]), key=lambda m: -(G[a][m]["weight"] + G[b][m]["weight"]))
        scored.append({"source": a, "target": b, "score": round(score, 4), "via": shared[:4]})
    scored.sort(key=lambda d: -d["score"])
    out["hidden_connections"] = scored[:limit]

    # Forgotten: important but not mentioned for a long time relative to the newest activity.
    dates = [_parse_date(n.get("last_seen")) for n in nodes.values()]
    dates = [d for d in dates if d]
    if dates:
        latest = max(dates)
        forgotten = []
        for node_id in specific:
            node = nodes[node_id]
            last = _parse_date(node.get("last_seen"))
            if not last or node.get("conversation_count", 0) < 2:
                continue
            idle_days = (latest - last).days
            if idle_days >= 60:
                forgotten.append({"node": node_id, "last_seen": node["last_seen"], "idle_days": idle_days,
                                  "score": round(node["importance"] * math.log1p(idle_days), 4)})
        forgotten.sort(key=lambda d: -d["score"])
        out["forgotten"] = forgotten[:limit]
    return out
