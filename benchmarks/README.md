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
| **graph_hybrid** | **0.673** | **0.757** | **0.820** | **0.698** | **0.580** | **0.599** |
| hybrid | 0.656 | 0.717 | 0.773 | 0.656 | 0.564 | 0.576 |
| bm25 | 0.640 | 0.717 | 0.765 | 0.657 | 0.572 | 0.581 |
| graph | 0.480 | 0.572 | 0.656 | 0.527 | 0.376 | 0.404 |
| dense | 0.468 | 0.544 | 0.641 | 0.485 | 0.401 | 0.411 |
| legacy (v0.1) | 0.463 | 0.538 | 0.561 | 0.480 | 0.394 | 0.407 |

Recall@10 by question type:

| mode | single-hop | multi-hop | temporal | open-domain |
|---|---|---|---|---|
| **graph_hybrid** | **0.901** | **0.459** | **0.747** | **0.367** |
| hybrid | 0.841 | 0.443 | 0.734 | 0.352 |
| bm25 | 0.850 | 0.435 | 0.724 | 0.333 |
| legacy (v0.1) | 0.630 | 0.304 | 0.579 | 0.263 |

What we learned on the way (dev split):

* Linking concepts only inside the conversation that proposed them left the graph with almost no
  cross-conversation edges — graph-only recall was 0.19. A global vocabulary linked against every
  conversation brought it to 0.60.
* Hub suppression matters: IDF-weighted seeds and NPMI edges keep names and generic words from flooding
  the walk.
* Static embeddings are a weak signal on this data, so fusion gives them a lower weight (`w_dense=0.25` for
  WordLlama; transformer embedders keep 1.0).
* A conversation-level prior (each chunk inherits the BM25 score of its whole conversation) added ~2 points
  of Recall@10 and MRR on both splits — evidence clusters inside sessions.
* Diversifying the top-k across conversations, graph-based query expansion (re-running BM25 with the
  labels of activated concepts) and semantic concept seeding with static embeddings all *hurt* on dev and
  are off by default (`max_per_conversation`, `w_expansion`, `semantic_seeds`).
* Time hints ("in June", "last spring") parsed from the question and used as a soft boost
  (`time_boost`): +8 pts R@10 on the 13 time-hinted dev questions, +1 pt on the 153 in test; neutral
  elsewhere. Applied in every mode so the comparison stays fair.
* Many LoCoMo "multi-hop" questions are really *category* questions ("What instruments does Melanie
  play?" → violin, clarinet) whose evidence never contains the category word. Term-level semantic seeds
  (`term_seeds`: each query word activates the concepts with the closest label embedding) gained +0.013 on
  dev but were neutral on test with static embeddings (R@20/MRR/multi-hop up, R@10 −0.004), so they are
  off by default — worth enabling and re-tuning with a transformer embedder.

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

### Without labels: LLM judge (`judged.py`)

`benchmarks/questions.json` holds 100 real questions about your own chats (from PR #9). No evidence labels
are needed: an LLM grades every retrieved passage 0–3 and the script reports graded **nDCG@10**,
**precision@10** (grade ≥ 2), **answerable@10** (some passage fully answers) and a head-to-head
**pairwise** comparison of two modes (asked twice with the order swapped to cancel position bias).

```bash
kb build                                   # index your real exports first
export GEMINI_API_KEY=...                  # rotates through flash-lite models to survive free-tier quotas
#   or any OpenAI-compatible endpoint (OpenAI, vLLM, Ollama, a Hugging Face Space):
export KB_JUDGE_BASE_URL=https://<host>/v1 KB_JUDGE_MODEL=<model> KB_JUDGE_API_KEY=...
python -m benchmarks.judged --vault ~/Documents/KnowledgeBase --delay 4
```

Grades are cached per (question, passage), so modes that retrieve the same passage are graded once and an
interrupted run resumes for free.

### With labels

The most useful benchmark is your own. Write 30–50 questions **from memory, the way you would actually
search** (copying wording from the chats makes every method look good), each labelled with the conversation
title(s) that answer it, in the format of `examples/sample_queries.jsonl`. Mix exact names ("GxEPD2"),
vague recall ("that thing about refresh flicker"), *when* questions and multi-hop questions. Then adapt
`benchmarks/sample.py` to point at your vault.
