"""Run the retrieval comparison: hybrid (graph + bge-small) vs baseline (bge-large).

Computes LLM-free structural + latency metrics for every question, and (if a
GEMINI_API_KEY is set) a Gemini pairwise judgment with A/B-order swapping.
Writes a timestamped JSON report and prints a readable summary.

Each question's result is written durably to results/progress.jsonl as it
completes, so an interrupted run (crash, lost connection, quota) loses nothing
and simply resumes where it left off on the next invocation.

Usage (from project root):
    python -m benchmarks.retrieval.run [--limit N] [--no-judge] [--device cpu|cuda] [--fresh]
"""
from __future__ import annotations
import argparse
import json
import os
import statistics
import threading
import time
from collections import defaultdict
from datetime import datetime

from . import config
from . import metrics as M
from .baseline import load_baseline_retriever
from .judge import Judge


def _rss_mb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return 0.0


class PeakRSSSampler:
    def __init__(self, interval=0.2):
        self.interval, self.peak_mb, self._running, self._thread = interval, 0.0, False, None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            self.peak_mb = max(self.peak_mb, _rss_mb())
            time.sleep(self.interval)

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)
        self.peak_mb = max(self.peak_mb, _rss_mb())
        return self.peak_mb


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


def load_hybrid():
    from src.retrieval.chunk_store import ChunkStore
    from src.retrieval.embedder import Embedder
    from src.retrieval.entity_index import EntityIndex
    from src.retrieval.graph_expander import GraphExpander
    from src.retrieval.retriever import Retriever
    from src.retrieval.vector_store import VectorStore

    embedder = Embedder()
    vector_store = VectorStore.load(config.INDEX_DIR / "embeddings.faiss", config.INDEX_DIR / "chunk_ids.txt")
    chunk_store = ChunkStore.load(config.INDEX_DIR / "chunks.json")
    entity_index = EntityIndex.load(config.INDEX_DIR / "entity_index.json")

    # graph_edges.json is an adjacency dict; GraphExpander.__init__ expects edge
    # dicts, so build it directly (same workaround as benchmarks/test_retreival.py).
    graph_expander = GraphExpander.__new__(GraphExpander)
    with open(config.INDEX_DIR / "graph_edges.json", "r", encoding="utf-8") as f:
        graph_expander._graph = defaultdict(list, json.load(f))

    retriever = Retriever(embedder, vector_store, chunk_store, entity_index, graph_expander)
    return retriever, embedder, vector_store, chunk_store


def _semantic_only(embedder, vector_store, chunk_store, query, top_k):
    query_embedding = embedder.embed_query(query)
    results = vector_store.search(query_embedding, k=top_k)
    ids = [cid for cid, _ in results]
    return chunk_store.get_many(ids)


def _pctl(values, p):
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((p / 100) * (len(ordered) - 1))))
    return ordered[idx]


def _load_progress(path):
    """Return {question_id: row} for questions already completed in a prior run."""
    done = {}
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            done[row["id"]] = row
        except (json.JSONDecodeError, KeyError):
            continue
    return done


def _append_progress(handle, row):
    """Append one result and force it to disk, so a crash can't lose it."""
    handle.write(json.dumps(row) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only first N questions")
    parser.add_argument("--no-judge", action="store_true", help="Skip the Gemini judge")
    parser.add_argument("--device", default="cuda", help="Device for the baseline embedder")
    parser.add_argument("--fresh", action="store_true", help="Ignore saved progress and start over")
    args = parser.parse_args()

    data = json.loads(config.QUESTIONS_PATH.read_text(encoding="utf-8"))
    questions = data["questions"][: args.limit] if args.limit else data["questions"]
    category_names = data.get("categories", {})

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    progress_path = config.RESULTS_DIR / "progress.jsonl"
    if args.fresh and progress_path.exists():
        progress_path.unlink()

    done = _load_progress(progress_path)
    remaining = [q for q in questions if q["id"] not in done]
    per_question = [done[q["id"]] for q in questions if q["id"] in done]
    if done:
        print(f"Resuming: {len(done)} already done, {len(remaining)} remaining.\n")

    peak_rss = 0.0
    if remaining:
        print("Loading hybrid retriever (bge-small + graph)...")
        hybrid, hy_embedder, hy_vs, hy_cs = load_hybrid()
        print(f"Loading baseline retriever ({config.BASELINE_EMBED_MODEL}) on {args.device}...")
        baseline = load_baseline_retriever(device=args.device)

        judge = None if args.no_judge else Judge()
        judging = judge is not None and judge.available
        print(f"Judge: {'Gemini pool ' + str(config.GEMINI_MODELS) if judging else 'DISABLED (no key / --no-judge)'}\n")

        sampler = PeakRSSSampler()
        sampler.start()

        with open(progress_path, "a", encoding="utf-8") as progress_fh:
            for i, q in enumerate(remaining, 1):
                qid, cat, text = q["id"], q["category"], q["question"]

                t0 = time.time()
                hybrid_chunks = hybrid.retrieve(text, top_k=config.TOP_K)
                hybrid_latency = time.time() - t0

                t0 = time.time()
                baseline_chunks = baseline.retrieve(text, top_k=config.TOP_K)
                baseline_latency = time.time() - t0

                hybrid_semantic = _semantic_only(hy_embedder, hy_vs, hy_cs, text, config.TOP_K)
                hybrid_ids = M.ids_of(hybrid_chunks)
                baseline_ids = M.ids_of(baseline_chunks)

                row = {
                    "id": qid, "category": cat, "question": text,
                    "jaccard": M.jaccard(hybrid_ids, baseline_ids),
                    "hybrid_unique": M.unique_to_first(hybrid_ids, baseline_ids),
                    "hybrid_source_diversity": M.source_diversity(hybrid_chunks),
                    "baseline_source_diversity": M.source_diversity(baseline_chunks),
                    "hybrid_cross_source": M.cross_source_rate(hybrid_chunks),
                    "baseline_cross_source": M.cross_source_rate(baseline_chunks),
                    "graph_contribution": M.graph_contribution(hybrid_ids, M.ids_of(hybrid_semantic)),
                    "hybrid_latency_s": hybrid_latency,
                    "baseline_latency_s": baseline_latency,
                }

                if judging:
                    r1 = judge.judge(text, hybrid_chunks, baseline_chunks)   # A = hybrid
                    time.sleep(config.JUDGE_DELAY_SECONDS)
                    r2 = judge.judge(text, baseline_chunks, hybrid_chunks)   # A = baseline
                    time.sleep(config.JUDGE_DELAY_SECONDS)
                    hybrid_score = (r1["score"] + (-r2["score"])) / 2        # + favors hybrid
                    row["judge_score"] = hybrid_score
                    row["judge_consistent"] = _sign(r1["score"]) == _sign(-r2["score"])
                    row["judge_verdicts"] = [r1["verdict"], r2["verdict"]]
                    row["judge_reasons"] = [r1["reason"], r2["reason"]]
                    row["judge_models"] = [r1["model"], r2["model"]]

                _append_progress(progress_fh, row)   # durable before we move on
                per_question.append(row)
                print(f"[{i}/{len(remaining)}] Q{qid} ({cat}) "
                      f"jac={row['jaccard']:.2f} hy_uniq={row['hybrid_unique']} "
                      f"graph+={row['graph_contribution']} "
                      f"hy_div={row['hybrid_source_diversity']} bl_div={row['baseline_source_diversity']}"
                      + (f" judge={row['judge_score']:+.1f}" if judging else ""), flush=True)

        peak_rss = sampler.stop()
    else:
        print("All questions already completed (use --fresh to redo).")

    per_question.sort(key=lambda r: r["id"])
    judging_for_agg = any("judge_score" in r for r in per_question)
    report = _aggregate(per_question, category_names, judging_for_agg, peak_rss)
    _print_summary(report, judging_for_agg)
    _save(report, per_question)

    # Full intended set finished — clear progress so the next run starts clean.
    if len(per_question) >= len(questions):
        progress_path.unlink(missing_ok=True)


def _mean(values):
    return statistics.mean(values) if values else 0.0


def _aggregate(rows, category_names, judging, peak_rss):
    def agg(subset):
        out = {
            "n": len(subset),
            "mean_jaccard": _mean([r["jaccard"] for r in subset]),
            "mean_hybrid_unique": _mean([r["hybrid_unique"] for r in subset]),
            "mean_graph_contribution": _mean([r["graph_contribution"] for r in subset]),
            "mean_hybrid_source_diversity": _mean([r["hybrid_source_diversity"] for r in subset]),
            "mean_baseline_source_diversity": _mean([r["baseline_source_diversity"] for r in subset]),
            "mean_hybrid_cross_source": _mean([r["hybrid_cross_source"] for r in subset]),
            "mean_baseline_cross_source": _mean([r["baseline_cross_source"] for r in subset]),
        }
        if judging:
            scores = [r["judge_score"] for r in subset if "judge_score" in r]
            out["mean_judge_score"] = _mean(scores)
            out["hybrid_win_rate"] = _mean([1 if s > 0 else 0 for s in scores])
            out["tie_rate"] = _mean([1 if s == 0 else 0 for s in scores])
            out["baseline_win_rate"] = _mean([1 if s < 0 else 0 for s in scores])
            out["judge_consistency"] = _mean([1 if r.get("judge_consistent") else 0 for r in subset if "judge_score" in r])
        return out

    by_category = {}
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["category"]].append(r)
    for cat, subset in sorted(grouped.items()):
        by_category[cat] = {"name": category_names.get(cat, cat), **agg(subset)}

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "models": {"hybrid_embed": config.HYBRID_EMBED_MODEL,
                   "baseline_embed": config.BASELINE_EMBED_MODEL,
                   "judge_pool": config.GEMINI_MODELS if judging else None},
        "top_k": config.TOP_K,
        "peak_rss_mb": round(peak_rss),
        "latency": {
            "hybrid_mean_s": _mean([r["hybrid_latency_s"] for r in rows]),
            "hybrid_p95_s": _pctl([r["hybrid_latency_s"] for r in rows], 95),
            "baseline_mean_s": _mean([r["baseline_latency_s"] for r in rows]),
            "baseline_p95_s": _pctl([r["baseline_latency_s"] for r in rows], 95),
        },
        "overall": agg(rows),
        "by_category": by_category,
    }


def _print_summary(report, judging):
    o = report["overall"]
    print("\n" + "=" * 66)
    print("RETRIEVAL COMPARISON  —  hybrid (graph+bge-small) vs baseline (bge-large)")
    print("=" * 66)
    print(f"Questions: {o['n']}   top_k: {report['top_k']}   peak RSS: {report['peak_rss_mb']} MB")
    lat = report["latency"]
    print(f"Latency (mean/p95):  hybrid {lat['hybrid_mean_s']:.3f}/{lat['hybrid_p95_s']:.3f}s   "
          f"baseline {lat['baseline_mean_s']:.3f}/{lat['baseline_p95_s']:.3f}s")
    print("\n-- Structural (no LLM) --")
    print(f"  mean top-10 Jaccard(hybrid,baseline): {o['mean_jaccard']:.2f}  (low = they differ a lot)")
    print(f"  mean hybrid-unique chunks vs baseline: {o['mean_hybrid_unique']:.1f}/10")
    print(f"  mean graph contribution (added over hybrid's own semantic): {o['mean_graph_contribution']:.1f}/10")
    print(f"  mean source diversity:  hybrid {o['mean_hybrid_source_diversity']:.1f}   baseline {o['mean_baseline_source_diversity']:.1f}")
    print(f"  mean cross-source rate: hybrid {o['mean_hybrid_cross_source']:.2f}   baseline {o['mean_baseline_cross_source']:.2f}")
    if judging:
        print("\n-- LLM judge (Gemini, A/B-swap averaged) --")
        print(f"  hybrid win / tie / loss: {o['hybrid_win_rate']:.0%} / {o['tie_rate']:.0%} / {o['baseline_win_rate']:.0%}")
        print(f"  mean preference score (-2..+2, + favors hybrid): {o['mean_judge_score']:+.2f}")
        print(f"  judge consistency under swap: {o['judge_consistency']:.0%}")
    print("\n-- By category --")
    for cat, c in report["by_category"].items():
        line = f"  {cat} {c['name'][:32]:<32} n={c['n']:<3} jac={c['mean_jaccard']:.2f} graph+={c['mean_graph_contribution']:.1f}"
        if judging:
            line += f" judge={c['mean_judge_score']:+.2f} win={c['hybrid_win_rate']:.0%}"
        print(line)
    print("=" * 66)


def _save(report, per_question):
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = config.RESULTS_DIR / f"comparison-{stamp}.json"
    path.write_text(json.dumps({"report": report, "per_question": per_question}, indent=2), encoding="utf-8")
    print(f"\nFull report saved: {path}")


if __name__ == "__main__":
    main()
