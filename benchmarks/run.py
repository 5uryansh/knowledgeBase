"""Benchmark runner.

    python -m benchmarks.run                         # LoCoMo, all modes, test split
    python -m benchmarks.run --split all --modes graph_hybrid,hybrid
    python -m benchmarks.run --embedder sentence-transformers   # bge-small (needs torch)

Each corpus is built once into ``benchmarks/.work/<dataset>/<embedder>/<id>`` and
reused (``--rebuild`` to force). Results are written to
``benchmarks/results/<dataset>_<embedder>_<split>.{json,md}``.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

from benchmarks.datasets import Corpus
from benchmarks.legacy import LegacyRetriever
from benchmarks.metrics import aggregate, score_query
from knowledgebase.config import Settings
from knowledgebase.enrichment.concepts import ConceptExtractor
from knowledgebase.parsers import PARSERS
from knowledgebase.pipeline import build
from knowledgebase.retrieval.embedders import build_embedder
from knowledgebase.retrieval.graph_retriever import GraphRetrieverConfig
from knowledgebase.retrieval.index import KnowledgeIndex
from knowledgebase.retrieval.retriever import MODES, FusionConfig

ROOT = Path(__file__).resolve().parent
SPLITS = {"dev": [0, 1], "test": [2, 3, 4, 5, 6, 7, 8, 9], "all": list(range(10))}
DEFAULT_MODES = ("bm25", "dense", "hybrid", "graph", "graph_hybrid", "legacy")


def resolve_evidence(corpus: Corpus, index: KnowledgeIndex) -> None:
    """Map evidence message locators to the chunk ids that contain them."""
    from knowledgebase.parsers.base import Conversation

    uid_of = {}
    for native_id, _ in corpus.evidence_locators.values():
        if native_id not in uid_of:
            uid_of[native_id] = Conversation(corpus.platform, native_id, "", None).uid
    groups: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for chunk in index.chunks:
        groups[chunk.conversation_id].append((chunk.turn, chunk.id))
    for g in groups.values():
        g.sort()

    def chunks_for(native_id: str, msg_idx: int) -> set[str]:
        g = groups.get(uid_of[native_id], [])
        start = max((t for t, _ in g if t <= msg_idx), default=None)
        return {cid for t, cid in g if t == start}

    for q in corpus.queries:
        resolved = []
        for ev in q.evidence:
            (dia,) = ev
            native_id, msg_idx = corpus.evidence_locators[dia]
            ids = chunks_for(native_id, msg_idx)
            if ids:
                resolved.append(ids)
        q.evidence = resolved
        q.evidence_sessions = {uid_of[s] for s in q.evidence_sessions}


def build_corpus(corpus: Corpus, work: Path, embedder, extractor, rebuild: bool) -> KnowledgeIndex:
    vault = work / corpus.id / "vault"
    settings = Settings(vault_dir=vault)
    if rebuild or not (settings.index_dir / "manifest.json").exists():
        conversations = list(PARSERS[corpus.platform](corpus.export_path).parse())
        report = build(settings, conversations=conversations, write_vault=False, embedder=embedder,
                       extractor=extractor, show_progress=False)
        print(f"  built {corpus.id}: {report}", flush=True)
    return KnowledgeIndex.load(settings.index_dir, embedder=embedder)


def evaluate(index: KnowledgeIndex, corpus: Corpus, modes: list[str], k_max: int = 20) -> list[dict]:
    rows = []
    legacy = LegacyRetriever(index) if "legacy" in modes else None
    for q in corpus.queries:
        if not q.evidence:
            continue
        for mode in modes:
            t0 = time.perf_counter()
            if mode == "legacy":
                ranked_idx = legacy.search(q.text)
                ranked = [index.chunks[i].id for i in ranked_idx]
                sessions = [index.chunks[i].conversation_id for i in ranked_idx]
            else:
                results = index.search(q.text, k=k_max, mode=mode)
                ranked = [r.chunk.id for r in results]
                sessions = [r.chunk.conversation_id for r in results]
            latency = (time.perf_counter() - t0) * 1000
            metrics = score_query(ranked, q.evidence, sessions, q.evidence_sessions)
            row = {"corpus": corpus.id, "query": q.id, "mode": mode, "category": q.category,
                   "metrics": metrics, "latency_ms": latency}
            if mode == "legacy":
                row["returned"] = len(ranked)
                everything = set(ranked)
                row["metrics"]["recall@all"] = sum(1 for ev in q.evidence if ev & everything) / len(q.evidence)
            rows.append(row)
    return rows


def to_markdown(summary: dict, modes: list[str], title: str, meta: dict) -> str:
    cats = ["all", "single-hop", "multi-hop", "temporal", "open-domain"]
    lines = [f"# {title}", "", "| setting | value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in meta.items()]
    lines += ["", "## Overall", "",
              "| mode | R@5 | R@10 | R@20 | complete@10 | MRR | nDCG@10 | session@5 | p50 ms |",
              "|---|---|---|---|---|---|---|---|---|"]
    for mode in modes:
        s = summary.get(f"{mode}|all")
        if not s:
            continue
        lines.append(
            f"| {mode} | {s['recall@5']:.3f} | {s['recall@10']:.3f} | {s['recall@20']:.3f} | "
            f"{s['complete@10']:.3f} | {s['mrr']:.3f} | {s['ndcg@10']:.3f} | {s.get('session@5', 0):.3f} | "
            f"{s['latency_p50_ms']:.1f} |"
        )
    lines += ["", "## Recall@10 by question type", "",
              "| mode | " + " | ".join(cats) + " |", "|---|" + "---|" * len(cats)]
    for mode in modes:
        cells = []
        for cat in cats:
            s = summary.get(f"{mode}|{cat}")
            cells.append(f"{s['recall@10']:.3f} (n={s['n']})" if s and cat != "all" else (f"{s['recall@10']:.3f}" if s else "–"))
        lines.append(f"| {mode} | " + " | ".join(cells) + " |")
    if "legacy" in modes and summary.get("legacy|all"):
        s = summary["legacy|all"]
        lines += ["", f"Legacy retriever returns **{s['avg_returned']:.0f} chunks per query on average** "
                      f"(top-10 = plain dense search, then an unranked graph tail); "
                      f"recall over its whole list: {s.get('recall@all', 0):.3f}."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="locomo", choices=["locomo"])
    ap.add_argument("--split", default="test", choices=list(SPLITS))
    ap.add_argument("--modes", default=",".join(DEFAULT_MODES))
    ap.add_argument("--embedder", default="auto", help="auto | sentence-transformers | wordllama | none")
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--entities", default="auto", help="auto | gliner | spacy")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--limit-queries", type=int, default=0, help="debug: only first N queries per corpus")
    ap.add_argument("--graph-config", default="", help="JSON overrides for GraphRetrieverConfig")
    ap.add_argument("--fusion-config", default="", help="JSON overrides for FusionConfig")
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    modes = [m for m in args.modes.split(",") if m]
    for m in modes:
        if m not in MODES and m != "legacy":
            ap.error(f"unknown mode {m}")

    embedder = build_embedder(args.embedder, args.embed_model)
    emb_tag = embedder.name.split(":")[0] if embedder else "nodense"
    if embedder and embedder.name.startswith("st:"):
        emb_tag = embedder.name.split("/")[-1]
    work = ROOT / ".work" / args.dataset / f"{emb_tag}-{args.entities}"
    settings = Settings(entity_extractor=args.entities)
    extractor = ConceptExtractor(settings)

    from benchmarks.datasets import locomo

    corpora = locomo.load(ROOT / "data" / "locomo10.json", work / "exports", SPLITS[args.split])
    gcfg = GraphRetrieverConfig(**json.loads(args.graph_config)) if args.graph_config else None
    fcfg = FusionConfig(**json.loads(args.fusion_config)) if args.fusion_config else None

    rows: list[dict] = []
    t0 = time.time()
    for corpus in corpora:
        index = build_corpus(corpus, work, embedder, extractor, args.rebuild)
        if gcfg or fcfg:
            index = KnowledgeIndex(index.chunks, index.graph, index.embedder, index.vectors, index.manifest, gcfg, fcfg)
        resolve_evidence(corpus, index)
        if args.limit_queries:
            corpus.queries = corpus.queries[: args.limit_queries]
        corpus_rows = evaluate(index, corpus, modes)
        rows.extend(corpus_rows)
        n_q = len({r['query'] for r in corpus_rows})
        print(f"  evaluated {corpus.id}: {n_q} queries", flush=True)

    summary = aggregate(rows)
    meta = {
        "dataset": f"LoCoMo ({args.split} split: samples {SPLITS[args.split]})",
        "queries": len({r["query"] for r in rows}),
        "embedder": embedder.name if embedder else "none",
        "entity extractor": extractor.entity_extractor.name,
        "graph config": json.dumps(gcfg.__dict__) if gcfg else "default",
        "fusion config": json.dumps(fcfg.__dict__) if fcfg else "default",
        "runtime": f"{time.time() - t0:.0f}s",
    }
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    stem = f"{args.dataset}_{emb_tag}_{args.split}" + (f"_{args.tag}" if args.tag else "")
    (out_dir / f"{stem}.json").write_text(json.dumps({"meta": meta, "summary": summary}, indent=1), encoding="utf-8")
    md = to_markdown(summary, modes, f"LoCoMo retrieval benchmark — {args.split}", meta)
    (out_dir / f"{stem}.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
