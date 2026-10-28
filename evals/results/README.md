# Evaluation artifacts

The files the top-level README cites. Superseded drafts, smoke runs and logs
were deleted; every number here can be reproduced with the scripts in
`scripts/`.

## Current

| artifact | what it shows |
|---|---|
| `retrieval_2026-10-28T23-28-42_gold100_dedup.json` | gold set, rerank on: recall@5 96/100, MRR 0.827, nDCG@10 0.859 |
| `retrieval_2026-10-28T23-38-32_gold100_norerank.json` | gold set, rerank off: recall@5 89/100, MRR 0.683, nDCG@10 0.745 |
| `answer_eval_2026-10-28T21-41-55_gold100_full.json` + `.md` | RAGAS over the gold set: faithfulness 0.913 (80/83), relevancy 0.803 (83/83), context precision 0.544 (83/83) |
| `answers_2026-10-28T21-41-55_gold100_full.json` | the 100 generated answers, contexts and per-query timings behind that run |
| `retrieval_2026-10-17T19-05-57_holdout.json` | held-out set: 12/12 recall@5, MRR 0.819 |
| `abstention_2026-10-17T20-50-24.json` | calibration negatives: 10/11 refused |
| `abstention_2026-10-17T21-26-55_holdout.json` | never-tuned negatives: 17/18 refused |
| `answer_eval_2026-10-18T00-33-57_strict_full.json` + `.md` | second-judge agreement, r = 0.322 (n = 12), on the earlier 23-question set |
| `eval_v2_2026-10-20T03-39-35_draft_norerank3.json` + `.md` | 120-question draft, rerank off: 80 answerable rows, 95.0% recall@5 |
| `eval_v2_2026-10-20T04-05-03_draft_rerank.json` + `.md` | same rows, rerank on: 92.5% recall@5 |
| `eval_v2_2026-10-20T03-44-01_draft_unanswerable_rerank2.json` + `.md` | the 20 unanswerable draft rows, scored for abstention rather than recall |

## Notes

- Retrieval metrics are document-level. Hits come back as chunks, so duplicates
  are collapsed to their first occurrence before scoring. Without that, five
  slots can hold three documents and recall@5 reads low.
- Each artifact records which provider and model actually served every answer
  and every judge call. On free tiers the requested pin is not a guarantee: in
  the gold run the judge pin served 31 of 917 calls, and Gemini fallbacks
  served the rest.
- The gold set is `evals/queries.jsonl` (100 questions, 107 gold documents).
  The held-out and negative probe sets are `evals/queries_holdout.jsonl`,
  `evals/queries_negative.jsonl` and `evals/queries_negative_holdout.jsonl`.

## Regenerating

```bash
python scripts/eval_retrieval.py --tag mine                 # gold set
python scripts/eval_retrieval.py --tag mine --no-rerank     # rerank off
python scripts/eval_answers.py --file evals/queries.jsonl --tag mine
```

`scripts/run_ragas_bg.ps1` runs the answer eval on two cores at low priority
and writes a log next to the results.
