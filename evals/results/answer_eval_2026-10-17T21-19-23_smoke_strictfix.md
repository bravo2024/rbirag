# RegIntel answer evaluation — 2026-10-17T21-19-23

Requested pins: answer `groq:openai/gpt-oss-120b` · judge `groq:openai/gpt-oss-120b` · judge 2 `gemini:gemini-3.7-flash` · strict pin: False · corpus `v1-53b832bb59` · k=6 · scored 3/3 answers (mode=llm).

**Served by** (pins are only requests; failover may serve other models):

- answers: {'groq/openai/gpt-oss-120b': 3}
- judge (all metrics): {'gemini/gemini-3.5-flash-lite': 28, 'groq/openai/gpt-oss-120b': 3, 'kilo-free/poolside/laguna-s-2.1:free': 3, 'kilo-free/openrouter/free': 1}
- judge 2 (faithfulness): {'gemini/gemini-3.5-flash-lite': 5, 'groq/openai/gpt-oss-120b': 1}

## Answer-level metrics (RAGAS)

| metric | mean | judged |
|---|---|---|
| faithfulness | 0.8 | 1/3 |
| answer_relevancy | 0.847 | 3/3 |
| llm_context_precision_without_reference | 0.861 | 3/3 |

## Pipeline mechanics

- modes: {'llm': 3, 'abstain': 0, 'extractive': 0}
- gold doc in retrieved context: 100.0%
- claims: 10 total, 3.33/answer, validator dropped 0, number-grounding flagged 1 (grounded rate 0.9)
- JSON parse retries (truncation guard): 0
- latency medians: retrieval 13739 ms · answer 0.02 s · end-to-end 13.79 s

## Judge agreement (faithfulness)

- not computed: same model served both judges (1/1 scored pairs)

## Per query

| # | mode | faith. | relev. | ctx prec. | gold rank | claims | e2e s | query |
|---|---|---|---|---|---|---|---|---|
| 1 | llm | None | 0.854 | 0.75 | 1 | 5 | 47.12 | periodic updation of KYC records for NBFC customers |
| 2 | llm | None | 0.8078 | 1.0 | 1 | 4 | 13.42 | periodic updation of KYC records for high risk customers of  |
| 3 | llm | 0.8 | 0.88 | 0.8333 | 1 | 1 | 13.79 | minimum capital adequacy ratio requirement for commercial ba |
