# LoCoMo retrieval benchmark — test

| setting | value |
|---|---|
| dataset | LoCoMo (test split: samples [2, 3, 4, 5, 6, 7, 8, 9]) |
| queries | 1304 |
| embedder | wordllama:l2_supercat_256 |
| entity extractor | spacy |
| graph config | default |
| fusion config | default |
| runtime | 29s |

## Overall

| mode | R@5 | R@10 | R@20 | complete@10 | MRR | nDCG@10 | session@5 | p50 ms |
|---|---|---|---|---|---|---|---|---|
| bm25 | 0.638 | 0.718 | 0.762 | 0.659 | 0.568 | 0.579 | 0.828 | 0.3 |
| dense | 0.463 | 0.538 | 0.633 | 0.480 | 0.397 | 0.407 | 0.700 | 0.3 |
| hybrid | 0.656 | 0.715 | 0.770 | 0.654 | 0.561 | 0.574 | 0.834 | 0.5 |
| graph | 0.470 | 0.564 | 0.653 | 0.519 | 0.371 | 0.398 | 0.643 | 1.0 |
| graph_hybrid | 0.673 | 0.755 | 0.817 | 0.696 | 0.578 | 0.598 | 0.829 | 1.8 |
| legacy | 0.463 | 0.538 | 0.561 | 0.480 | 0.394 | 0.407 | 0.700 | 0.4 |

## Recall@10 by question type

| mode | all | single-hop | multi-hop | temporal | open-domain |
|---|---|---|---|---|---|
| bm25 | 0.718 | 0.850 (n=727) | 0.435 (n=239) | 0.730 (n=257) | 0.333 (n=81) |
| dense | 0.538 | 0.630 (n=727) | 0.304 (n=239) | 0.579 (n=257) | 0.263 (n=81) |
| hybrid | 0.715 | 0.840 (n=727) | 0.443 (n=239) | 0.736 (n=257) | 0.333 (n=81) |
| graph | 0.564 | 0.683 (n=727) | 0.304 (n=239) | 0.553 (n=257) | 0.295 (n=81) |
| graph_hybrid | 0.755 | 0.900 (n=727) | 0.459 (n=239) | 0.747 (n=257) | 0.361 (n=81) |
| legacy | 0.538 | 0.630 (n=727) | 0.304 (n=239) | 0.579 (n=257) | 0.263 (n=81) |

Legacy retriever returns **334 chunks per query on average** (top-10 = plain dense search, then an unranked graph tail); recall over its whole list: 1.000.
