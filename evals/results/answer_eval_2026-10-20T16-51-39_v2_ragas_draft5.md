# RegIntel answer evaluation — 2026-10-20T16-51-39

Requested pins: answer `groq:openai/gpt-oss-120b` · judge `groq:openai/gpt-oss-120b` · judge 2 `gemini:gemini-3.7-flash` · strict pin: False · corpus `v1-182406aedd` · k=6 · scored 5/5 answers (mode=llm).

**Served by** (pins are only requests; failover may serve other models):

- answers: {'groq/openai/gpt-oss-120b': 4, 'gemini/gemini-3.5-flash': 1}
- judge (all metrics): {'groq/openai/gpt-oss-120b': 52, 'gemini/gemini-3.6-flash': 3}
- judge 2 (faithfulness): {'gemini/gemini-3.6-flash': 6, 'gemini/gemini-3.5-flash': 4}

## Answer-level metrics (RAGAS)

| metric | mean | judged |
|---|---|---|
| faithfulness | 0.928 | 5/5 |
| answer_relevancy | 0.809 | 5/5 |
| llm_context_precision_without_reference | 0.785 | 5/5 |

## Pipeline mechanics

- modes: {'llm': 5, 'abstain': 0, 'extractive': 0}
- gold doc in retrieved context: 100.0%
- claims: 12 total, 2.4/answer, validator dropped 0, number-grounding flagged 4 (grounded rate 0.667)
- JSON parse retries (truncation guard): 0
- latency medians: retrieval 8167 ms · answer 5.24 s · end-to-end 16.11 s

## Judge agreement (faithfulness)

- second judge `gemini:gemini-3.7-flash`: n=5 (different model served each judge), mean |Δ| = 0.068, Pearson r = 0.298

## Per query

| # | mode | faith. | relev. | ctx prec. | gold rank | claims | e2e s | query |
|---|---|---|---|---|---|---|---|---|
| 1 | llm | 0.8333 | 0.8654 | 1.0 | 1 | 2 | 45.87 | periodic updation of KYC records for high-risk customers of  |
| 2 | llm | 1.0 | 0.8416 | 0.25 | 1 | 2 | 12.21 | interest rate offered on savings deposits by commercial bank |
| 3 | llm | 0.9 | 0.7964 | 0.8767 | 1 | 2 | 11.88 | benchmark rate and spread for interest rates on advances |
| 4 | llm | 1.0 | 0.8164 | 0.8333 | 1 | 2 | 16.11 | cash reserve ratio and statutory liquidity ratio maintenance |
| 5 | llm | 0.9091 | 0.724 | 0.9667 | 1 | 4 | 135.24 | classification and valuation of the investment portfolio inc |
