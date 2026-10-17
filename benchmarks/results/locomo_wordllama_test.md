# LoCoMo retrieval benchmark — test

| setting | value |
|---|---|
| dataset | LoCoMo (test split: samples [2, 3, 4, 5, 6, 7, 8, 9]) |
| queries | 1304 |
| embedder | wordllama:l2_supercat_256 |
| entity extractor | spacy |
| graph config | default |
| fusion config | default |
| runtime | 10s |

## Overall

| mode | R@5 | R@10 | R@20 | complete@10 | MRR | nDCG@10 | session@5 | p50 ms |
|---|---|---|---|---|---|---|---|---|
| bm25 | 0.640 | 0.717 | 0.765 | 0.657 | 0.572 | 0.581 | 0.827 | 0.4 |
| dense | 0.468 | 0.544 | 0.641 | 0.485 | 0.401 | 0.411 | 0.710 | 0.4 |
| hybrid | 0.656 | 0.717 | 0.773 | 0.656 | 0.564 | 0.576 | 0.833 | 0.7 |
| graph | 0.480 | 0.572 | 0.656 | 0.527 | 0.376 | 0.404 | 0.656 | 1.5 |
| graph_hybrid | 0.673 | 0.757 | 0.820 | 0.698 | 0.580 | 0.599 | 0.825 | 2.7 |
| legacy | 0.463 | 0.538 | 0.561 | 0.480 | 0.394 | 0.407 | 0.700 | 0.5 |

## Recall@10 by question type

| mode | all | single-hop | multi-hop | temporal | open-domain |
|---|---|---|---|---|---|
| bm25 | 0.717 | 0.850 (n=727) | 0.435 (n=239) | 0.724 (n=257) | 0.333 (n=81) |
| dense | 0.544 | 0.639 (n=727) | 0.306 (n=239) | 0.581 (n=257) | 0.275 (n=81) |
| hybrid | 0.717 | 0.841 (n=727) | 0.443 (n=239) | 0.734 (n=257) | 0.352 (n=81) |
| graph | 0.572 | 0.693 (n=727) | 0.304 (n=239) | 0.566 (n=257) | 0.301 (n=81) |
| graph_hybrid | 0.757 | 0.901 (n=727) | 0.459 (n=239) | 0.747 (n=257) | 0.367 (n=81) |
| legacy | 0.538 | 0.630 (n=727) | 0.304 (n=239) | 0.579 (n=257) | 0.263 (n=81) |

Legacy retriever returns **334 chunks per query on average** (top-10 = plain dense search, then an unranked graph tail); recall over its whole list: 1.000.
