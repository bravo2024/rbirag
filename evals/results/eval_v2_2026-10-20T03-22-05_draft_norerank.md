# Eval set v2 — 2026-10-20T03-22-05

Set: `eval_set_v2.jsonl` · 120 questions · k=5 · rerank=off

## Summary

**retrieval**

| metric | value |
|---|---|
| n | 80 |
| recall@5 | 0.887 |
| mrr@5 | 0.742 |
| recall@10 | 0.925 |
| ndcg@10 | 0.769 |

**retrieval_by_institution**

| metric | value |
|---|---|
| bank | {'n': 50, 'recall@5': 0.9} |
| nbfc | {'n': 30, 'recall@5': 0.867} |

**as_of**

| metric | value |
|---|---|
| n | 20 |
| pass_rate | 0.95 |
| false_include | 0 |
| false_exclude | 1 |

**unanswerable**

| metric | value |
|---|---|
| n | 20 |
| scored | 0 |
| abstain_rate | None |

## Failures / reviewed flags

| id | category | query | gold | got |
|---|---|---|---|---|
| v2-a15 | answerable | capital charge for credit risk under the standardised approach | md13399 | md13159, md13137, md13729, md13159, md13705 |
| v2-a17 | answerable | credit valuation adjustment framework for derivative exposures | md13729 | md13159, md13399, md13159, md13159, md13159 |
| v2-a39 | answerable | credit risk management framework for NBFCs | md12954 | md12949, md13590, md12949, md12949, md13592 |
| v2-a40 | answerable | credit facilities and loans by NBFCs | md12957 | md12942, md12948, md12965, md12946, md12948 |
| v2-a43 | answerable | governance framework for NBFCs | md12960 | md13592, md12936, md13592, md13587, md12933 |
| v2-a59 | answerable | credit information reporting by NBFCs | md12955 | md12946, md12946, md12957, md13592, md13590 |
| v2-t12 | as_of | credit valuation adjustment framework for derivative exposures | md13729 | md13159, md13399, md13159, md13159, md13159 |
| v2-m07 | amendment | What change was made to non-option derivative contract limits for small finance  | md13105, nn13741 | nn13740, md13136, md12682, md10485, md10485 |
| v2-m16 | amendment | What is the reference number of the Small Finance Banks Miscellaneous Amendment  | nn13741 | nn13731, nn13740, nn13730, nn13732, md12799 |
| v2-m18 | amendment | What paragraph of the Payments Banks Capital Adequacy Directions was amended in  | md13097, nn13734 | nn13732, nn13731, nn13730, nn13740, md13729 |
