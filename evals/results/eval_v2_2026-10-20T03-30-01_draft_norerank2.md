# Eval set v2 — 2026-10-20T03-30-01

Set: `eval_set_v2.jsonl` · 120 questions · k=5 · rerank=off

## Summary

**retrieval**

| metric | value |
|---|---|
| n | 80 |
| recall@5 | 0.95 |
| mrr@5 | 0.795 |
| recall@10 | 0.975 |
| ndcg@10 | 0.816 |

**retrieval_by_institution**

| metric | value |
|---|---|
| bank | {'n': 47, 'recall@5': 1.0} |
| nbfc | {'n': 30, 'recall@5': 0.867} |
| all | {'n': 3, 'recall@5': 1.0} |

**as_of**

| metric | value |
|---|---|
| n | 20 |
| pass_rate | 1.0 |
| false_include | 0 |
| false_exclude | 0 |

**unanswerable**

| metric | value |
|---|---|
| n | 20 |
| scored | 0 |
| abstain_rate | None |

## Failures / reviewed flags

| id | category | query | gold | got |
|---|---|---|---|---|
| v2-a39 | answerable | credit risk management framework for NBFCs | md12954 | md12949, md13590, md12949, md12949, md13592 |
| v2-a40 | answerable | credit facilities and loans by NBFCs | md12957 | md12942, md12948, md12965, md12946, md12948 |
| v2-a43 | answerable | governance framework for NBFCs | md12960 | md13592, md12936, md13592, md13587, md12933 |
| v2-a59 | answerable | credit information reporting by NBFCs | md12955 | md12946, md12946, md12957, md13592, md13590 |
