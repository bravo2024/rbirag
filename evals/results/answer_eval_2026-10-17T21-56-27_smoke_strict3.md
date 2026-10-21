# RegIntel answer evaluation — 2026-10-17T21-56-27

Requested pins: answer `gemini:gemini-3.5-flash-lite` · judge `nvidia-build:nvidia/nemotron-3-super-120b-a12b` · judge 2 `gemini:gemini-3.5-flash-lite` · strict pin: True · corpus `v1-53b832bb59` · k=6 · scored 2/2 answers (mode=llm).

**Served by** (pins are only requests; failover may serve other models):

- answers: {'gemini/gemini-3.5-flash-lite': 2}
- judge (all metrics): {'nvidia-build/nvidia/nemotron-3-super-120b-a12b': 4}
- judge 2 (faithfulness): {'gemini/gemini-3.5-flash-lite': 4}

## Answer-level metrics (RAGAS)

| metric | mean | judged |
|---|---|---|
| faithfulness | 0.8 | 2/2 |

## Pipeline mechanics

- modes: {'llm': 2, 'abstain': 0, 'extractive': 0}
- gold doc in retrieved context: 100.0%
- claims: 7 total, 3.5/answer, validator dropped 0, number-grounding flagged 1 (grounded rate 0.857)
- JSON parse retries (truncation guard): 0
- latency medians: retrieval 31380 ms · answer 4.72 s · end-to-end 36.1 s

## Judge agreement (faithfulness)

- second judge `gemini:gemini-3.5-flash-lite`: n=2 (different model served each judge), mean |Δ| = 0.133, Pearson r = 1.0

## Per query

| # | mode | faith. | relev. | ctx prec. | gold rank | claims | e2e s | query |
|---|---|---|---|---|---|---|---|---|
| 1 | llm | 1.0 | None | None | 1 | 4 | 36.1 | periodic updation of KYC records for NBFC customers |
| 2 | llm | 0.6 | None | None | 1 | 3 | 8.44 | periodic updation of KYC records for high risk customers of  |
