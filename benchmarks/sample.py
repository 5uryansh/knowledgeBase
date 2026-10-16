"""Mini benchmark on the bundled *technical* sample chats (examples/sample_queries.jsonl).

    python -m benchmarks.sample --embedder wordllama

25 hand-written queries in the way people actually search their history
("vLLM out of memory at startup", "which quantization did I use both for the
microcontroller and for LLM serving"), each labelled with the conversation(s)
that answer it. Relevance is judged per conversation. Tiny, so treat it as a
sanity check next to LoCoMo — and as a template for writing your own.
"""
from __future__ import annotations

import argparse
import json
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path

from benchmarks.legacy import LegacyRetriever
from knowledgebase.config import Settings
from knowledgebase.pipeline import build
from knowledgebase.retrieval.embedders import build_embedder
from knowledgebase.retrieval.index import KnowledgeIndex
from knowledgebase.retrieval.retriever import MODES

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "examples" / "sample_exports"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="auto")
    ap.add_argument("--entities", default="auto")
    args = ap.parse_args()

    embedder = build_embedder(args.embedder)
    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(vault_dir=Path(tmp), entity_extractor=args.entities)
        settings.exports = {
            "chatgpt": str(SAMPLES / "chatgpt" / "conversations.json"),
            "claude": str(SAMPLES / "claude" / "conversations.json"),
            "gemini": str(SAMPLES / "gemini" / "MyActivity.json"),
            "claude-code": str(SAMPLES / "claude-code" / "projects"),
        }
        build(settings, write_vault=False, embedder=embedder, show_progress=False)
        index = KnowledgeIndex.load(settings.index_dir, embedder=embedder)

    rows = [json.loads(l) for l in (ROOT / "examples" / "sample_queries.jsonl").read_text().splitlines() if l.strip()]
    legacy = LegacyRetriever(index)
    modes = [m for m in MODES if m != "dense" or embedder] + (["legacy"] if embedder else [])
    scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    def relevant(title: str, rels: list[str]) -> bool:
        return any(title.startswith(r[:40]) or r[:40] in title for r in rels)

    for row in rows:
        for mode in modes:
            if mode == "legacy":
                titles = [index.chunks[i].title for i in legacy.search(row["query"])][:10]
            else:
                titles = [r.chunk.title for r in index.search(row["query"], k=10, mode=mode)]
            convs = list(dict.fromkeys(titles))  # conversation-level ranking
            found = {r for r in row["relevant"] if any(relevant(t, [r]) for t in convs[:5])}
            first = next((i for i, t in enumerate(convs, 1) if relevant(t, row["relevant"])), None)
            for cat in ("all", row["category"]):
                scores[mode][f"{cat}:recall@5"].append(len(found) / len(row["relevant"]))
                scores[mode][f"{cat}:mrr"].append(1 / first if first else 0.0)

    lines = ["| mode | Recall@5 (all) | Recall@5 (multi-hop) | MRR |", "|---|---|---|---|"]
    for mode in modes:
        s = scores[mode]
        lines.append(f"| {mode} | {statistics.fmean(s['all:recall@5']):.3f} | "
                     f"{statistics.fmean(s['multi-hop:recall@5']):.3f} | {statistics.fmean(s['all:mrr']):.3f} |")
    md = (f"# Sample technical chats — {len(rows)} hand-written queries\n\nembedder: "
          f"{embedder.name if embedder else 'none'}\n\n" + "\n".join(lines) + "\n")
    out = ROOT / "benchmarks" / "results" / f"sample_{(embedder.name.split(':')[0] if embedder else 'nodense')}.md"
    out.write_text(md)
    print(md)


if __name__ == "__main__":
    main()
