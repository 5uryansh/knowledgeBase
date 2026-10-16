# LoCoMo retrieval benchmark — test

| setting | value |
|---|---|
| dataset | LoCoMo (test split: samples [2, 3, 4, 5, 6, 7, 8, 9]) |
| queries | 1304 |
| embedder | wordllama:l2_supercat_256 |
| entity extractor | spacy |
| graph config | default |
| fusion config | default |
| runtime | 42s |

## Overall

| mode | R@5 | R@10 | R@20 | complete@10 | MRR | nDCG@10 | session@5 | p50 ms |
|---|---|---|---|---|---|---|---|---|
| bm25 | 0.638 | 0.718 | 0.762 | 0.659 | 0.568 | 0.579 | 0.828 | 0.4 |
| dense | 0.465 | 0.540 | 0.636 | 0.484 | 0.399 | 0.409 | 0.699 | 0.4 |
| hybrid | 0.657 | 0.713 | 0.770 | 0.652 | 0.563 | 0.574 | 0.835 | 0.6 |
| graph | 0.478 | 0.576 | 0.666 | 0.531 | 0.379 | 0.406 | 0.660 | 1.1 |
| graph_hybrid | 0.657 | 0.736 | 0.799 | 0.676 | 0.561 | 0.580 | 0.827 | 1.6 |
| legacy | 0.465 | 0.540 | 0.563 | 0.484 | 0.397 | 0.409 | 0.699 | 0.4 |

## Recall@10 by question type

| mode | all | single-hop | multi-hop | temporal | open-domain |
|---|---|---|---|---|---|
| bm25 | 0.718 | 0.850 (n=727) | 0.435 (n=239) | 0.730 (n=257) | 0.333 (n=81) |
| dense | 0.540 | 0.630 (n=727) | 0.305 (n=239) | 0.591 (n=257) | 0.263 (n=81) |
| hybrid | 0.713 | 0.838 (n=727) | 0.441 (n=239) | 0.731 (n=257) | 0.333 (n=81) |
| graph | 0.576 | 0.694 (n=727) | 0.318 (n=239) | 0.579 (n=257) | 0.277 (n=81) |
| graph_hybrid | 0.736 | 0.868 (n=727) | 0.467 (n=239) | 0.734 (n=257) | 0.357 (n=81) |
| legacy | 0.540 | 0.630 (n=727) | 0.305 (n=239) | 0.591 (n=257) | 0.263 (n=81) |

Legacy retriever returns **334 chunks per query on average** (top-10 = plain dense search, then an unranked graph tail); recall over its whole list: 1.000.
