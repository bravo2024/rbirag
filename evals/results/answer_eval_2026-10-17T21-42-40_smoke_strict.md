# RegIntel answer evaluation — 2026-10-17T21-42-40

Requested pins: answer `groq:openai/gpt-oss-120b` · judge `groq:openai/gpt-oss-120b` · judge 2 `gemini:gemini-3.5-flash-lite` · strict pin: True · corpus `v1-53b832bb59` · k=6 · scored 0/2 answers (mode=llm).

**Served by** (pins are only requests; failover may serve other models):

- answers: {}
- judge (all metrics): {}
- judge 2 (faithfulness): {}

## Answer-level metrics (RAGAS)

| metric | mean | judged |
|---|---|---|

## Pipeline mechanics

- modes: {'llm': 0, 'abstain': 0, 'extractive': 2}
- gold doc in retrieved context: 100.0%
- claims: 0 total, 0.0/answer, validator dropped 0, number-grounding flagged 0 (grounded rate None)
- JSON parse retries (truncation guard): 0
- latency medians: retrieval 29564 ms · answer 56.64 s · end-to-end 67.63 s

## Per query

| # | mode | faith. | relev. | ctx prec. | gold rank | claims | e2e s | query |
|---|---|---|---|---|---|---|---|---|
| 1 | extractive | None | None | None | 1 | 0 | 40.52 | periodic updation of KYC records for NBFC customers |
| 2 | extractive | None | None | None | 1 | 0 | 67.63 | periodic updation of KYC records for high risk customers of  |
