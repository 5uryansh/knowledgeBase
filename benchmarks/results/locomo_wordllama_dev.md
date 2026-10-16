# LoCoMo retrieval benchmark — dev

| setting | value |
|---|---|
| dataset | LoCoMo (dev split: samples [0, 1]) |
| queries | 231 |
| embedder | wordllama:l2_supercat_256 |
| entity extractor | spacy |
| graph config | default |
| fusion config | default |
| runtime | 1s |

## Overall

| mode | R@5 | R@10 | R@20 | complete@10 | MRR | nDCG@10 | session@5 | p50 ms |
|---|---|---|---|---|---|---|---|---|
| bm25 | 0.661 | 0.742 | 0.787 | 0.688 | 0.585 | 0.602 | 0.847 | 0.3 |
| dense | 0.462 | 0.544 | 0.665 | 0.498 | 0.366 | 0.390 | 0.665 | 0.3 |
| hybrid | 0.680 | 0.737 | 0.799 | 0.680 | 0.568 | 0.590 | 0.858 | 0.4 |
| graph | 0.519 | 0.621 | 0.682 | 0.589 | 0.405 | 0.444 | 0.691 | 0.8 |
| graph_hybrid | 0.662 | 0.776 | 0.839 | 0.719 | 0.578 | 0.609 | 0.853 | 1.2 |
| legacy | 0.462 | 0.544 | 0.597 | 0.498 | 0.365 | 0.390 | 0.665 | 0.3 |

## Recall@10 by question type

| mode | all | single-hop | multi-hop | temporal | open-domain |
|---|---|---|---|---|---|
| bm25 | 0.742 | 0.811 (n=114) | 0.403 (n=43) | 0.921 (n=63) | 0.318 (n=11) |
| dense | 0.544 | 0.579 (n=114) | 0.203 (n=43) | 0.762 (n=63) | 0.273 (n=11) |
| hybrid | 0.737 | 0.803 (n=114) | 0.403 (n=43) | 0.921 (n=63) | 0.318 (n=11) |
| graph | 0.621 | 0.693 (n=114) | 0.290 (n=43) | 0.778 (n=63) | 0.273 (n=11) |
| graph_hybrid | 0.776 | 0.846 (n=114) | 0.438 (n=43) | 0.937 (n=63) | 0.455 (n=11) |
| legacy | 0.544 | 0.579 (n=114) | 0.203 (n=43) | 0.762 (n=63) | 0.273 (n=11) |

Legacy retriever returns **207 chunks per query on average** (top-10 = plain dense search, then an unranked graph tail); recall over its whole list: 1.000.
