# Encoder A/B — 2026-10-20T01:55:10

Corpus: 398 docs / 14186 chunks (`v1-182406aedd`)

| model | encoder | set | method | n | recall@5 | recall@10 | MRR@5 | nDCG@10 |
|---|---|---|---|---|---|---|---|---|
| a | `BAAI/bge-small-en-v1.5` | queries | dense | 23 | 0.870 | 0.913 | 0.768 | 0.794 |
| a | `BAAI/bge-small-en-v1.5` | queries | hybrid | 23 | 0.913 | 1.000 | 0.797 | 0.816 |
| a | `BAAI/bge-small-en-v1.5` | queries_holdout | dense | 12 | 0.917 | 0.917 | 0.736 | 0.783 |
| a | `BAAI/bge-small-en-v1.5` | queries_holdout | hybrid | 12 | 0.917 | 1.000 | 0.688 | 0.776 |
| b | `vivekkopthsd/hinglish-st-33m` | queries | dense | 23 | 0.739 | 0.870 | 0.570 | 0.626 |
| b | `vivekkopthsd/hinglish-st-33m` | queries | hybrid | 23 | 0.913 | 1.000 | 0.717 | 0.756 |
| b | `vivekkopthsd/hinglish-st-33m` | queries_holdout | dense | 12 | 0.833 | 0.833 | 0.561 | 0.628 |
| b | `vivekkopthsd/hinglish-st-33m` | queries_holdout | hybrid | 12 | 0.750 | 0.833 | 0.653 | 0.707 |
