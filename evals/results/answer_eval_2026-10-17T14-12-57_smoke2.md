# RegIntel answer evaluation — 2026-10-17T14-12-57

Answer model `groq:openai/gpt-oss-120b` · judge `gemini:gemini-3.7-flash` · corpus `v1-53b832bb59` · k=6 · scored 2/2 answers (mode=llm).

## Answer-level metrics (RAGAS)

| metric | mean | judged |
|---|---|---|
| faithfulness | 0.933 | 1/2 |
| answer_relevancy | 0.869 | 2/2 |
| llm_context_precision_without_reference | 0.792 | 2/2 |

## Pipeline mechanics

- modes: {'llm': 2, 'abstain': 0, 'extractive': 0}
- gold doc in retrieved context: 100.0%
- claims: 9 total, 4.5/answer, validator dropped 0, number-grounding flagged 1 (grounded rate 0.889)
- JSON parse retries (truncation guard): 0
- latency medians: retrieval 31972 ms · answer 4.55 s · end-to-end 31.98 s

## Judge agreement (faithfulness)

- second judge `groq:openai/gpt-oss-120b`: n=1, mean |Δ| = 0.067, Pearson r = None

## Per query

| # | mode | faith. | relev. | ctx prec. | gold rank | claims | e2e s | query |
|---|---|---|---|---|---|---|---|---|
| 1 | llm | None | 0.8725 | 0.5833 | 1 | 5 | 31.98 | periodic updation of KYC records for NBFC customers |
| 2 | llm | 0.9333 | 0.8663 | 1.0 | 1 | 4 | 18.62 | periodic updation of KYC records for high risk customers of  |

_Judge models seen in cache during scoring: {'gemini-2.5-flash': 15, 'gemini-2.5-flash-lite': 5}_
