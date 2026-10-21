# RegIntel answer evaluation — 2026-10-20T21-37-44

Requested pins: answer `groq:openai/gpt-oss-120b` · judge `gemini:gemini-3.5-flash-lite -> gemini:gemini-2.5-flash-lite -> kilo-free:nvidia/nemotron-3-ultra-550b-a55b:free -> kilo-free:nvidia/nemotron-3.5-lightning:free -> kilo-free:dots-studio/dots-3-note-preview:free -> kilo-free:cohere/north-mini-code:free -> kilo-free:poolside/laguna-s-2.1:free` · judge 2 `gemini:gemini-3.7-flash` · strict pin: False · context 1024 tok/total · corpus `v1-182406aedd` · k=6 · scored 2/2 answers (mode=llm).

**Served by** (pins are only requests; failover may serve other models):

- answers: {'groq/openai/gpt-oss-120b': 2}
- judge (all metrics): {'gemini/gemini-3.5-flash-lite': 12, 'kilo-free/nvidia/nemotron-3-ultra-550b-a55b:free': 4}
- judge 2 (faithfulness): {}

## Answer-level metrics (RAGAS)

| metric | mean | judged |
|---|---|---|
| faithfulness | 0.5 | 2/2 |
| answer_relevancy | 0.816 | 2/2 |
| llm_context_precision_without_reference | 0.5 | 2/2 |

## Pipeline mechanics

- modes: {'llm': 2, 'abstain': 0, 'extractive': 0}
- gold doc in retrieved context: 100.0%
- claims: 4 total, 2.0/answer, validator dropped 0, number-grounding flagged 0 (grounded rate 1.0)
- JSON parse retries (truncation guard): 0
- latency medians: retrieval 37147 ms · answer 6.43 s · end-to-end 43.6 s

## Per query

| # | mode | faith. | relev. | ctx prec. | gold rank | claims | e2e s | query |
|---|---|---|---|---|---|---|---|---|
| 1 | llm | 1.0 | 0.8154 | 1.0 | 1 | 2 | 43.6 | periodic updation of KYC records for high-risk customers of  |
| 2 | llm | 0.0 | 0.8161 | 0.0 | 1 | 2 | 11.93 | interest rate offered on savings deposits by commercial bank |
