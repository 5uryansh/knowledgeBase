# Sample technical chats — 25 hand-written queries

embedder: wordllama:l2_supercat_256

| mode | Recall@5 (all) | Recall@5 (multi-hop) | MRR |
|---|---|---|---|
| bm25 | 0.953 | 0.870 | 0.953 |
| dense | 0.953 | 0.870 | 0.960 |
| hybrid | 0.953 | 0.870 | 0.953 |
| graph | 0.947 | 0.852 | 0.907 |
| graph_hybrid | 0.953 | 0.870 | 0.960 |
| legacy | 0.953 | 0.870 | 0.960 |
