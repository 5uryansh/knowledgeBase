"""Absolute IR metrics from graded relevance.

Given a system's ranked passages (as relevance grades 0-3 in rank order) and the
grades of the whole pooled candidate set, compute Precision@k, pooled Recall@k,
Hit@k, MRR and nDCG@k. Relevance is binary at grade >= RELEVANT_THRESHOLD for
the set metrics; nDCG uses the graded gains directly.
"""
from __future__ import annotations
import math

RELEVANT_THRESHOLD = 2   # 0 irrelevant, 1 marginal, 2 relevant, 3 directly answers


def _is_relevant(grade: int) -> bool:
    return grade >= RELEVANT_THRESHOLD


def precision_at_k(ranked_grades, k) -> float:
    return sum(1 for g in ranked_grades[:k] if _is_relevant(g)) / k


def recall_at_k(ranked_grades, k, total_relevant) -> float:
    if total_relevant == 0:
        return 0.0
    return sum(1 for g in ranked_grades[:k] if _is_relevant(g)) / total_relevant


def hit_at_k(ranked_grades, k) -> float:
    return 1.0 if any(_is_relevant(g) for g in ranked_grades[:k]) else 0.0


def mrr(ranked_grades) -> float:
    for i, g in enumerate(ranked_grades, start=1):
        if _is_relevant(g):
            return 1.0 / i
    return 0.0


def _dcg(grades, k) -> float:
    return sum((2 ** g - 1) / math.log2(i + 1) for i, g in enumerate(grades[:k], start=1))


def ndcg_at_k(ranked_grades, k, pool_grades) -> float:
    ideal = sorted(pool_grades, reverse=True)
    idcg = _dcg(ideal, k)
    if idcg == 0:
        return 0.0
    return _dcg(ranked_grades, k) / idcg


def compute(ranked_grades, pool_grades) -> dict:
    """All metrics for one system's ranked top-k, given the pool's grades."""
    total_relevant = sum(1 for g in pool_grades if _is_relevant(g))
    return {
        "precision_at_10": precision_at_k(ranked_grades, 10),
        "recall_at_1": recall_at_k(ranked_grades, 1, total_relevant),
        "recall_at_5": recall_at_k(ranked_grades, 5, total_relevant),
        "recall_at_10": recall_at_k(ranked_grades, 10, total_relevant),
        "hit_at_1": hit_at_k(ranked_grades, 1),
        "hit_at_5": hit_at_k(ranked_grades, 5),
        "hit_at_10": hit_at_k(ranked_grades, 10),
        "mrr": mrr(ranked_grades),
        "ndcg_at_5": ndcg_at_k(ranked_grades, 5, pool_grades),
        "ndcg_at_10": ndcg_at_k(ranked_grades, 10, pool_grades),
    }
