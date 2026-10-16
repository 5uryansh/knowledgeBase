# Benchmarks

Two questions drive every change to retrieval:

1. **Does it find the right passage?** (recall, ranking quality)
2. **Does the graph actually help, or just add noise?** (ablations: the same index queried with and without
   the graph signal)

## Datasets

### LoCoMo (main benchmark)

[LoCoMo](https://github.com/snap-research/locomo) — *Evaluating Very Long-Term Conversational Memory of LLM
Agents* (Maharana et al., ACL 2024). 10 chat histories, each 19–32 sessions (~600 turns), with ~1,500
answerable questions whose **evidence turns are annotated**. Questions are typed:

| type | what it tests |
|---|---|
| single-hop | one fact from one turn |
| multi-hop | facts spread over several turns/sessions — where a graph should help |
| temporal | *when* something happened |
| open-domain | needs inference beyond the text |

`benchmarks/datasets/locomo.py` writes every session as a conversation in a **Claude.ai export file**, so
the benchmark exercises the real parser → concept extraction → graph → index pipeline. Adversarial
(unanswerable) questions are skipped.

Splits: **dev** = histories 0–1 (used for tuning), **test** = histories 2–9 (reported). Never tune on test.

### Sample technical chats (sanity check)

`examples/sample_queries.jsonl` — 25 hand-written queries over the 36 bundled sample conversations (GPU,
RAG, ESP32, web dev, agents). With only 36 conversations every method saturates, so it is a smoke test and
a template, not a leaderboard.

## Metrics

Computed per question over the ranked chunk list (`metrics.py`):

* **Recall@k** — share of evidence items covered by the top-k chunks
* **complete@k** — 1 if *all* evidence is covered (the real bar for multi-hop questions)
* **MRR** — 1 / rank of the first relevant chunk
* **nDCG@10** — rank-aware relevance of the top 10
* **session@k** — share of the evidence conversations that appear in the top k
* **latency** p50 / p95 per query

## Retrieval modes compared

| mode | signals |
|---|---|
| `bm25` | lexical |
| `dense` | embeddings |
| `hybrid` | BM25 + dense, RRF |
| `graph` | Personalized PageRank seeded only by concepts named in the query |
| `graph_hybrid` | BM25 + dense, then PPR seeded by the query and the first-stage hits, all fused (default) |
| `legacy` | the v0.1 retriever, re-implemented faithfully (`legacy.py`) |

## Results (LoCoMo test split, 1,304 questions)

Environment: WordLlama embeddings + spaCy entities (`lite` install). Full tables with per-type breakdowns
are in [`results/`](results/).

| mode | R@5 | R@10 | R@20 | complete@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|---|
| **graph_hybrid** | **0.657** | **0.736** | **0.799** | **0.676** | 0.561 | **0.580** |
| hybrid | 0.657 | 0.713 | 0.770 | 0.652 | 0.563 | 0.574 |
| bm25 | 0.638 | 0.718 | 0.762 | 0.659 | 0.568 | 0.579 |
| graph | 0.478 | 0.576 | 0.666 | 0.531 | 0.379 | 0.406 |
| dense | 0.465 | 0.540 | 0.636 | 0.484 | 0.399 | 0.409 |
| legacy (v0.1) | 0.465 | 0.540 | 0.563 | 0.484 | 0.397 | 0.409 |

Recall@10 by question type:

| mode | single-hop | multi-hop | temporal | open-domain |
|---|---|---|---|---|
| **graph_hybrid** | **0.868** | **0.467** | **0.734** | **0.357** |
| hybrid | 0.838 | 0.441 | 0.731 | 0.333 |
| bm25 | 0.850 | 0.435 | 0.730 | 0.333 |
| legacy (v0.1) | 0.630 | 0.305 | 0.591 | 0.263 |

What we learned on the way (dev split):

* Linking concepts only inside the conversation that proposed them left the graph with almost no
  cross-conversation edges — graph-only recall was 0.19. A global vocabulary linked against every
  conversation brought it to 0.60.
* Hub suppression matters: IDF-weighted seeds and NPMI edges keep names and generic words from flooding
  the walk.
* Static embeddings are a weak signal on this data, so fusion gives them a lower weight (`w_dense=0.25` for
  WordLlama; transformer embedders keep 1.0).
* Diversifying the top-k across conversations and semantic concept seeding with static embeddings both
  *hurt* and are off by default.

## Run it

```bash
python -m benchmarks.run --split test                       # all modes, writes results/locomo_<embedder>_test.*
python -m benchmarks.run --split dev --modes graph_hybrid,hybrid --rebuild
python -m benchmarks.run --embedder sentence-transformers   # bge-small (needs the ml extra)
python -m benchmarks.tune --trials 120                      # random/local search on the dev split
python -m benchmarks.sample                                 # sample technical chats
```

Pass `--graph-config '{"damping": 0.85}'` or `--fusion-config '{"w_graph": 2}'` to try settings without
editing code.

## Benchmark your own history

The most useful benchmark is your own. Write 30–50 questions **from memory, the way you would actually
search** (copying wording from the chats makes every method look good), each labelled with the conversation
title(s) that answer it, in the format of `examples/sample_queries.jsonl`. Mix exact names ("GxEPD2"),
vague recall ("that thing about refresh flicker"), *when* questions and multi-hop questions. Then adapt
`benchmarks/sample.py` to point at your vault.
