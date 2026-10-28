# Eval set v2 — 120-question draft

`evals/eval_set_v2.jsonl` is a **draft** evaluation set built to test the three
things the original 23-question set did not cover: point-in-time retrieval,
amendment provenance, and abstention. Every row starts as
`"review_status": "draft"`. **These are not verified gold labels yet.** A domain
reviewer must check the questions and gold documents before the numbers are
quoted as a benchmark.

## Composition

| category | n | what it tests | gold |
| --- | ---: | --- | --- |
| `answerable` | 60 | natural questions over current Master Directions (30 commercial-bank, 30 NBFC) | one or two document ids |
| `as_of` | 20 | ten documents × (before effective date, on/after it); does the date filter include exactly the right version? | probe document id |
| `amendment` | 20 | the October 2026 RSS notifications and the Directions they amend | notification + parent |
| `unanswerable` | 20 | out-of-regulator and in-domain traps; the pipeline should abstain | none |

The 60 answerable questions are deliberately spread across themes (KYC, credit
risk, capital, governance, audit, outsourcing, cyber, cards, ALM, registration,
microfinance, P2P, account aggregator, etc.) rather than concentrated in one
chapter. The questions were written from document titles and notification text,
not selected because a retriever already ranked the gold document first. After
the first dry run, a few queries were made more specific for clarity (adding the
entity or an acronym such as “CVA”); no gold document was changed to match
retrieval output. The four retrieval misses described below were left as-is.

## Row schema

```json
{
  "id": "v2-a01",
  "category": "answerable",
  "query": "periodic updation of KYC records for high-risk customers of commercial banks",
  "institution": "bank",
  "as_of": null,
  "gold_doc_ids": ["md13141"],
  "expected_topic": "KYC periodic updation frequency and risk categories",
  "expected": "answer",
  "review_status": "draft",
  "reviewer_notes": ""
}
```

Additional fields for specific categories:

* `as_of`: `probe_doc_id`, `expect_present` (bool), and `as_of` (ISO date).
* `unanswerable`: `expected = "abstain"` and `expected_topic` names the reason.

`institution` is `bank` (commercial banks), `nbfc` (NBFC/HFC group), or `null`
(no entity filter).

## What the harness does

`scripts/eval_v2.py` runs each row through the product engine and reports:

* **answerable / amendment** — document-level recall@k, MRR@k, recall@10 and
  nDCG@10, the same metrics as `scripts/eval_retrieval.py`.
* **as_of** — whether the probe document appears in the top 10 exactly when
  expected, **plus** a `future_leaks` check: no returned document may have an
  `effective_from` later than the requested date. This is the real date-filter
  test; checking only the probe would miss a different future version leaking
  in.
* **unanswerable** — retrieval-level abstention. This needs rerank scores; run
  it without `--no-rerank`.

```bash
python scripts/eval_v2.py --no-rerank --tag draft_norerank
python scripts/eval_v2.py --only unanswerable --tag draft_unanswerable
python scripts/eval_v2.py --only answerable --k 5 --tag answerable_rerank
python scripts/eval_v2.py --reviewed-only --tag reviewed
python -m pytest tests/test_eval_set_v2.py -q
```

Results are written to `evals/results/eval_v2_<timestamp>_<tag>.{json,md}`.

## Draft baseline (2026-10-20, no rerank for retrieval; rerank for abstention)

| slice | result |
| --- | --- |
| answerable + amendment, n=80, rerank off | recall@5 **.950**, MRR@5 **.795**, recall@10 **.975**, nDCG@10 **.816** |
| answerable + amendment, n=80, rerank on | recall@5 **.925**, MRR@5 **.812**, recall@10 **.963**, nDCG@10 **.826** |
| — commercial banks, n=47 (off / on) | recall@5 **1.000 / .979** |
| — NBFC, n=30 (off / on) | recall@5 **.867 / .833** |
| — no institution filter, n=3 (off / on) | recall@5 **1.000 / 1.000** |
| as_of, n=20 | **20/20** pass; 0 false includes, 0 false excludes, **0 future leaks** |
| unanswerable, n=20 | abstention **18/20 = .900** |

Artifacts:
`evals/results/eval_v2_2026-10-20T03-39-35_draft_norerank3.{json,md}`,
`evals/results/eval_v2_2026-10-20T04-05-03_draft_rerank.{json,md}` and
`evals/results/eval_v2_2026-10-20T03-44-01_draft_unanswerable_rerank2.{json,md}`.

Rerank does not rescue any of the four misses and adds two top-5 misses:
`v2-a18` (board governance) and `v2-a34` (NBFC income recognition/asset
classification), with gold at rank 10 and rank 8 respectively (recall@10 hits). It improves MRR/nDCG
slightly but lowers recall@5 from .950 to .925. The 23-question set showed
rerank as a clear win; this broader draft does not. Do not change the default
from draft labels alone — review the set, then investigate rerank depth and the
generic-query failure mode.

### Answer-level RAGAS smoke (draft)

`scripts/eval_answers.py` now accepts this file directly:

```bash
python scripts/eval_answers.py --file evals/eval_set_v2.jsonl \
  --category answerable --limit 5 --tag v2_ragas_draft5
```

After review, add `--reviewed-only` to score only verified rows.

On the **first five answerable commercial-bank rows** (draft, not reviewed):

| metric | mean | judged |
| --- | --- | --- |
| faithfulness | **0.928** | 5/5 |
| answer_relevancy | **0.809** | 5/5 |
| llm_context_precision_without_reference | **0.785** | 5/5 |

All five were LLM mode and the gold document was in context 5/5. The citation
validator dropped 0 claims, but number-grounding flagged 4 of 12 (grounded rate
0.667), all four on the investment-portfolio answer. Faithfulness second-judge
agreement was n=5 (mean |Δ| 0.068, r=0.298), too small to interpret. This is a
wiring/quality smoke, not a full-set RAGAS benchmark: the rows are all
commercial-bank, the labels are draft, and the remaining 55 answerable + 20
amendment rows have not been scored. One answer fell back to
`gemini/gemini-3.5-flash` (pin `groq/openai/gpt-oss-120b`), so this is a
mixed-model smoke. Artifact:
`evals/results/answer_eval_2026-10-20T16-51-39_v2_ragas_draft5.{json,md}`.

The four retrieval misses are all generic NBFC queries:
`v2-a39` (credit risk management framework), `v2-a40` (credit facilities),
`v2-a43` (governance framework), `v2-a59` (credit information reporting).
`v2-a39` and `v2-a59` are at rank 7 (recall@10 hits); `v2-a40` and `v2-a43` are
outside the top 10. This is a real weakness worth investigating: the title
appears in the preliminary passage, not throughout the document, and generic
topic queries match several sibling directions. Do **not** rename the questions
to make them pass; that would hide the finding.

The two abstention misses are `v2-u01` (SEBI disclosure) and `v2-u13` (October
2026 RBI press release). In both cases the reranker scored a topically nearby
RBI passage above +2.0: `v2-u01` retrieves a direction that *mentions* SEBI
ICDR, and `v2-u13` retrieves a Master Direction preamble that mentions the
Statement on Development and Regulatory Policies. Neither passage actually
answers the question. This is the expected failure mode of a logit gate that
scores relevance, not answerability.

## Review protocol

1. Open `evals/eval_set_v2_review.csv` (Excel-friendly, UTF-8 BOM). It includes
   `gold_titles`, so you can review without looking up every id. The JSONL is
   the harness input; the CSV is a snapshot for review.
2. For each row:
   * confirm the question is answerable from the corpus (or genuinely not);
   * confirm the `institution` filter does not exclude the gold document;
   * confirm `gold_doc_ids` — open the document, not just the search result;
   * for `as_of`, confirm the effective date and the expected include/exclude;
   * for `amendment`, confirm the notification actually amends the parent and
     that the question is not asking for a detail the notification does not
     state.
3. Set `review_status` to `reviewed` and record doubts in `reviewer_notes`.
   The harness reports all rows and, separately, the reviewed-only subset once
   at least one row is reviewed.
4. Re-run `scripts/eval_v2.py`; only reviewed rows should be quoted as a
   benchmark.

**Do not re-run `scripts/build_eval_set_v2.py` after review starts** — it
overwrites the JSONL and CSV. The generator is provenance for the initial
draft; use `--force` only if you intend to discard review edits.

## Known limitations

* The 60 answerable rows are **document-level** gold. They do not yet carry
  verified chunk/paragraph ids, so the harness measures retrieval, not whether
  the final answer cites the right clause. Paragraph-level gold is the next
  step and should be added during review.
* The `as_of` rows test one probe per date, not every document in force on that
  date. They use the default non-strict `as_of` filter (undated text is kept);
  `strict_as_of` behaviour is covered by the unit tests, not this set. They are
  a date-filter regression test, not a full point-in-time benchmark.
* The unanswerable set mixes out-of-regulator questions with in-domain traps.
  A regulator-specific model may reasonably answer some of them from general
  knowledge; the product's intended behaviour is still to abstain when the
  corpus does not support the answer. These 20 rows are scored by the
  abstention harness, not by RAGAS; the RAGAS smoke covers 5 of the 100
  answer-bearing rows (60 answerable + 20 amendment + 20 as-of).
* All labels are draft. Do not compare these numbers to the verified 23-question
  gold set as if they were the same kind of measurement.
