"""Retrieval metrics over ranked chunk lists.

For a query with evidence items E (each satisfiable by one or more chunk ids):

* ``recall@k``      — share of evidence items covered by the top-k chunks
* ``complete@k``    — 1 if *every* evidence item is covered (the bar for multi-hop)
* ``mrr``           — 1 / rank of the first chunk that covers any evidence (rank ≤ 100)
* ``ndcg@10``       — binary relevance, ideal = min(#relevant, 10)
* ``session@k``     — share of evidence conversations that appear in the top-k
"""
from __future__ import annotations

import math
import statistics
from collections import defaultdict

KS = (1, 5, 10, 20)


def score_query(ranked: list[str], evidence: list[set[str]], ranked_sessions: list[str] | None = None,
                evidence_sessions: set[str] | None = None) -> dict[str, float]:
    relevant = set().union(*evidence) if evidence else set()
    out: dict[str, float] = {}
    for k in KS:
        top = set(ranked[:k])
        covered = sum(1 for ev in evidence if ev & top)
        out[f"recall@{k}"] = covered / len(evidence) if evidence else 0.0
        out[f"complete@{k}"] = 1.0 if evidence and covered == len(evidence) else 0.0
        if ranked_sessions is not None and evidence_sessions:
            out[f"session@{k}"] = len(set(ranked_sessions[:k]) & evidence_sessions) / len(evidence_sessions)
    rr = 0.0
    for r, cid in enumerate(ranked[:100], start=1):
        if cid in relevant:
            rr = 1.0 / r
            break
    out["mrr"] = rr
    dcg = sum(1.0 / math.log2(r + 1) for r, cid in enumerate(ranked[:10], start=1) if cid in relevant)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(relevant), 10) + 1))
    out["ndcg@10"] = dcg / idcg if idcg else 0.0
    return out


def aggregate(rows: list[dict]) -> dict[str, dict[str, float]]:
    """rows: [{"mode", "category", "metrics": {...}, "latency_ms"}] -> {mode: {metric: mean}} incl. per category."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["mode"], "all")].append(row)
        groups[(row["mode"], row["category"])].append(row)
    summary: dict[str, dict[str, float]] = {}
    for (mode, cat), items in groups.items():
        metrics = defaultdict(list)
        for it in items:
            for name, value in it["metrics"].items():
                metrics[name].append(value)
        entry = {name: statistics.fmean(values) for name, values in metrics.items()}
        lat = sorted(it["latency_ms"] for it in items)
        entry["latency_p50_ms"] = lat[len(lat) // 2]
        entry["latency_p95_ms"] = lat[min(len(lat) - 1, int(len(lat) * 0.95))]
        entry["n"] = len(items)
        if "returned" in items[0]:
            entry["avg_returned"] = statistics.fmean(it["returned"] for it in items)
        summary[f"{mode}|{cat}"] = entry
    return summary
