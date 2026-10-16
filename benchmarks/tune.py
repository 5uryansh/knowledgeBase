"""Hyper-parameter search for fusion + graph settings on the *dev* split only.

    python -m benchmarks.tune --embedder wordllama --trials 60

Test-split numbers in the README are produced with the parameters found here,
never tuned on test.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics

from benchmarks.datasets import locomo
from benchmarks.metrics import score_query
from benchmarks.run import ROOT, SPLITS, build_corpus, resolve_evidence
from knowledgebase.config import Settings
from knowledgebase.enrichment.concepts import ConceptExtractor
from knowledgebase.retrieval.embedders import build_embedder
from knowledgebase.retrieval.graph_retriever import GraphRetrieverConfig
from knowledgebase.retrieval.index import KnowledgeIndex
from knowledgebase.retrieval.retriever import FusionConfig

SPACE = {
    "w_bm25": [1.0],
    "w_dense": [0.0, 0.25, 0.5, 0.75, 1.0],
    "w_graph": [0.5, 0.75, 1.0, 1.5, 2.0],
    "rrf_k": [10, 30, 60],
    "first_stage_k": [5, 10, 20, 40],
    "damping": [0.3, 0.5, 0.7, 0.85],
    "chunk_seed_weight": [0.0, 0.25, 0.5, 1.0, 2.0],
    "prf_concept_weight": [0.0, 0.3, 1.0],
    "concept_concept": [0.0, 0.5, 1.0, 2.0],
    "chunk_chunk": [0.0, 0.1, 0.3, 1.0],
    "semantic_seeds": [0, 3, 5, 10],
    "semantic_seed_weight": [0.25, 0.5, 1.0, 2.0],
    "semantic_min_sim": [0.2, 0.3, 0.4, 0.5],
}
FUSION_KEYS = {"w_bm25", "w_dense", "w_graph", "rrf_k", "first_stage_k", "max_per_conversation"}


def objective(rows: list[dict]) -> float:
    return statistics.fmean(0.5 * r["recall@10"] + 0.25 * r["ndcg@10"] + 0.25 * r["mrr"] for r in rows)


def evaluate(corpora, indexes, params: dict) -> float:
    fcfg = FusionConfig(**{k: v for k, v in params.items() if k in FUSION_KEYS})
    gcfg = GraphRetrieverConfig(**{k: v for k, v in params.items() if k not in FUSION_KEYS})
    rows = []
    for corpus, base in zip(corpora, indexes):
        index = base.with_config(gcfg, fcfg)
        for q in corpus.queries:
            if q.evidence:
                ranked = [r.chunk.id for r in index.search(q.text, k=20, mode="graph_hybrid")]
                rows.append(score_query(ranked, q.evidence))
    return objective(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="auto")
    ap.add_argument("--entities", default="auto")
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    embedder = build_embedder(args.embedder)
    emb_tag = embedder.name.split(":")[0] if embedder else "nodense"
    if embedder and embedder.name.startswith("st:"):
        emb_tag = embedder.name.split("/")[-1]
    work = ROOT / ".work" / "locomo" / f"{emb_tag}-{args.entities}"
    extractor = ConceptExtractor(Settings(entity_extractor=args.entities))
    corpora = locomo.load(ROOT / "data" / "locomo10.json", work / "exports", SPLITS["dev"])
    indexes = []
    for corpus in corpora:
        index = build_corpus(corpus, work, embedder, extractor, rebuild=False)
        resolve_evidence(corpus, index)
        indexes.append(index)

    defaults = {**FusionConfig().__dict__, **GraphRetrieverConfig().__dict__}
    defaults = {k: v for k, v in defaults.items() if k in SPACE}
    best = (evaluate(corpora, indexes, defaults), defaults)
    print(f"default: {best[0]:.4f}")
    rng = random.Random(args.seed)
    for t in range(args.trials):
        # Mostly local moves around the incumbent, sometimes a random restart.
        if rng.random() < 0.3:
            params = {k: rng.choice(v) for k, v in SPACE.items()}
        else:
            params = dict(best[1])
            for k in rng.sample(list(SPACE), 2):
                params[k] = rng.choice(SPACE[k])
        score = evaluate(corpora, indexes, params)
        if score > best[0]:
            best = (score, params)
            print(f"[{t}] new best {score:.4f}: {json.dumps(params)}", flush=True)
    print("BEST", json.dumps(best[1]))
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"tuned_{emb_tag}.json").write_text(json.dumps({"score": best[0], "params": best[1]}, indent=1))


if __name__ == "__main__":
    main()
