import json

import pytest
from conftest import ROOT

from benchmarks.metrics import score_query
from knowledgebase.retrieval.bm25 import BM25Index
from knowledgebase.retrieval.retriever import MODES


def test_bm25_ranks_exact_terms_first():
    idx = BM25Index().fit(["vLLM out of memory at startup", "FAISS index types", "PagedAttention in vLLM"])
    scores = idx.scores("vllm memory")
    assert scores.argmax() == 0
    assert scores[1] == 0


def test_metrics():
    m = score_query(["x", "a", "y", "b"], [{"a"}, {"b"}])
    assert m["recall@1"] == 0 and m["recall@5"] == 1.0
    assert m["mrr"] == 0.5
    assert m["complete@5"] == 1.0 and m["complete@1"] == 0.0


def test_index_files_and_chunks(index, demo_vault):
    files = {p.name for p in (demo_vault / "indexes").iterdir()}
    assert {"chunks.jsonl", "graph.json", "manifest.json", "graph_edges.json", "graph_relations.json",
            "entity_index.json"} <= files
    assert len(index.chunks) > 50
    assert all(c.conversation_id and c.title for c in index.chunks)


@pytest.mark.parametrize("mode", [m for m in MODES if m != "dense"])
def test_every_mode_returns_ranked_results(index, mode):
    results = index.search("vLLM out of memory", k=5, mode=mode)
    assert results, mode
    assert [r.rank for r in results] == list(range(1, len(results) + 1))


def test_sample_queries_find_the_right_conversation(index):
    """Hand-written queries over the sample data: graph_hybrid should find most of them in the top 5."""
    rows = [json.loads(l) for l in (ROOT / "examples" / "sample_queries.jsonl").read_text().splitlines() if l.strip()]
    hits = 0
    for row in rows:
        titles = {r.chunk.title for r in index.search(row["query"], k=5)}
        hits += any(any(t.startswith(rel[:40]) or rel[:40] in t for t in titles) for rel in row["relevant"])
    assert hits / len(rows) >= 0.8, f"only {hits}/{len(rows)} sample queries answered in top 5"


def test_graph_explains_results(index):
    results = index.search("how did I fix the e-ink ghosting", k=3)
    assert results[0].chunk.title.startswith("ESP32 e-paper weather station")
    assert any(r.via for r in results)
