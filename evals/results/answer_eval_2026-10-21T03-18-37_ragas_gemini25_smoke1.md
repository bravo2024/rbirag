# RegIntel answer evaluation — 2026-10-21T03-18-37

Requested pins: answer `groq:openai/gpt-oss-120b` · judge `gemini:gemini-2.5-flash -> gemini:gemini-3.5-flash-lite` · judge 2 `gemini:gemini-3.7-flash` · strict pin: False · context 1024 tok/total · corpus `v1-182406aedd` · k=6 · scored 1/1 answers (mode=llm).

**Served by** (pins are only requests; failover may serve other models):

- answers: {'groq/openai/gpt-oss-120b': 1}
- judge (all metrics): {'gemini/gemini-3.5-flash-lite': 2}
- judge 2 (faithfulness): {}

## Answer-level metrics (RAGAS)

| metric | mean | judged |
|---|---|---|
| faithfulness | 1.0 | 1/1 |

## Pipeline mechanics

- modes: {'llm': 1, 'abstain': 0, 'extractive': 0}
- gold doc in retrieved context: 100.0%
- claims: 2 total, 2.0/answer, validator dropped 0, number-grounding flagged 0 (grounded rate 1.0)
- JSON parse retries (truncation guard): 0
- latency medians: retrieval 37147 ms · answer 6.43 s · end-to-end 43.6 s

## Per query

| # | mode | faith. | relev. | ctx prec. | gold rank | claims | e2e s | query |
|---|---|---|---|---|---|---|---|---|
| 1 | llm | 1.0 | None | None | 1 | 2 | 43.6 | periodic updation of KYC records for high-risk customers of  |
