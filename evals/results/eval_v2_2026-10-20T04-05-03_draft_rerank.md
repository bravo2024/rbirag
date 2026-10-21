# Eval set v2 — 2026-10-20T04-05-03

Set: `eval_set_v2.jsonl` · 80 questions · k=5 · rerank=on

## Summary

**retrieval**

| metric | value |
|---|---|
| n | 80 |
| recall@5 | 0.925 |
| mrr@5 | 0.812 |
| recall@10 | 0.963 |
| ndcg@10 | 0.826 |

**retrieval_by_institution**

| metric | value |
|---|---|
| bank | {'n': 47, 'recall@5': 0.979} |
| nbfc | {'n': 30, 'recall@5': 0.833} |
| all | {'n': 3, 'recall@5': 1.0} |

## Failures / reviewed flags

| id | category | query | gold | got |
|---|---|---|---|---|
| v2-a18 | answerable | board governance and responsibilities of commercial banks | md13161 | md13162, md13645, md13641, md13638, md13147 |
| v2-a34 | answerable | income recognition and asset classification for NBFCs | md12948 | md12947, md12931, md12957, md12947, md12945 |
| v2-a39 | answerable | credit risk management framework for NBFCs | md12954 | md13590, md12949, md12949, md12941, md13592 |
| v2-a40 | answerable | credit facilities and loans by NBFCs | md12957 | md13590, md12954, md12933, md12942, md12948 |
| v2-a43 | answerable | governance framework for NBFCs | md12960 | md13592, md13592, md13590, md12945, md13587 |
| v2-a59 | answerable | credit information reporting by NBFCs | md12955 | md12946, md12957, md12946, md12952, md12957 |
