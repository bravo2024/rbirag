# RegIntel answer evaluation — 2026-10-17T21-14-08

Requested pins: answer `groq:openai/gpt-oss-120b` · judge `groq:openai/gpt-oss-120b` · judge 2 `gemini:gemini-3.7-flash` · strict pin: False · corpus `v1-53b832bb59` · k=6 · scored 3/3 answers (mode=llm).

**Served by** (pins are only requests; failover may serve other models):

- answers: {'groq/openai/gpt-oss-120b': 3}
- judge (all metrics): {'gemini/gemini-3.5-flash-lite': 30, 'groq/openai/gpt-oss-120b': 3, 'kilo-free/poolside/laguna-s-2.1:free': 2}
- judge 2 (faithfulness): {'gemini/gemini-3.5-flash-lite': 5, 'groq/openai/gpt-oss-120b': 1}

## Answer-level metrics (RAGAS)

| metric | mean | judged |
|---|---|---|
| faithfulness | 1.0 | 2/3 |
| answer_relevancy | 0.852 | 3/3 |
| llm_context_precision_without_reference | 0.861 | 3/3 |

## Pipeline mechanics

- modes: {'llm': 3, 'abstain': 0, 'extractive': 0}
- gold doc in retrieved context: 100.0%
- claims: 10 total, 3.33/answer, validator dropped 0, number-grounding flagged 1 (grounded rate 0.9)
- JSON parse retries (truncation guard): 0
- latency medians: retrieval 13904 ms · answer 3.33 s · end-to-end 17.27 s

## Judge agreement (faithfulness)

- not computed: same model served both judges (2/2 scored pairs)

## Per query

| # | mode | faith. | relev. | ctx prec. | gold rank | claims | e2e s | query |
|---|---|---|---|---|---|---|---|---|
| 1 | llm | None | 0.8691 | 0.75 | 1 | 5 | 48.23 | periodic updation of KYC records for NBFC customers |
| 2 | llm | 1.0 | 0.8078 | 1.0 | 1 | 4 | 12.88 | periodic updation of KYC records for high risk customers of  |
| 3 | llm | 1.0 | 0.88 | 0.8333 | 1 | 1 | 17.27 | minimum capital adequacy ratio requirement for commercial ba |
