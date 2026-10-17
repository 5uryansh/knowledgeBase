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
| bm25 | 0.661 | 0.742 | 0.787 | 0.688 | 0.588 | 0.605 | 0.847 | 0.4 |
| dense | 0.457 | 0.551 | 0.660 | 0.506 | 0.366 | 0.393 | 0.669 | 0.4 |
| hybrid | 0.685 | 0.742 | 0.799 | 0.684 | 0.571 | 0.595 | 0.858 | 0.6 |
| graph | 0.524 | 0.626 | 0.687 | 0.589 | 0.412 | 0.451 | 0.706 | 1.3 |
| graph_hybrid | 0.690 | 0.783 | 0.846 | 0.727 | 0.602 | 0.627 | 0.847 | 2.3 |
| legacy | 0.457 | 0.546 | 0.600 | 0.502 | 0.358 | 0.386 | 0.665 | 0.4 |

## Recall@10 by question type

| mode | all | single-hop | multi-hop | temporal | open-domain |
|---|---|---|---|---|---|
| bm25 | 0.742 | 0.811 (n=114) | 0.403 (n=43) | 0.921 (n=63) | 0.318 (n=11) |
| dense | 0.551 | 0.605 (n=114) | 0.191 (n=43) | 0.746 (n=63) | 0.273 (n=11) |
| hybrid | 0.742 | 0.803 (n=114) | 0.403 (n=43) | 0.937 (n=63) | 0.318 (n=11) |
| graph | 0.626 | 0.684 (n=114) | 0.316 (n=43) | 0.794 (n=63) | 0.273 (n=11) |
| graph_hybrid | 0.783 | 0.855 (n=114) | 0.426 (n=43) | 0.952 (n=63) | 0.455 (n=11) |
| legacy | 0.546 | 0.596 (n=114) | 0.191 (n=43) | 0.746 (n=63) | 0.273 (n=11) |

Legacy retriever returns **207 chunks per query on average** (top-10 = plain dense search, then an unranked graph tail); recall over its whole list: 1.000.
