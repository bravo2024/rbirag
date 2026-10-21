# RegIntel answer evaluation — 2026-10-18T00-33-57

Requested pins: answer `nvidia-build:nvidia/nemotron-3-super-120b-a12b` · judge `gemini:gemini-3.5-flash-lite` · judge 2 `kilo-free:poolside/laguna-s-2.1:free` · strict pin: True · corpus `v1-53b832bb59` · k=6 · scored 19/23 answers (mode=llm).

**Served by** (pins are only requests; failover may serve other models):

- answers: {'nvidia-build/nvidia/nemotron-3-super-120b-a12b': 19}
- judge (all metrics): {'gemini/gemini-3.5-flash-lite': 118}
- judge 2 (faithfulness): {'kilo-free/poolside/laguna-s-2.1:free': 51}

## Answer-level metrics (RAGAS)

| metric | mean | judged |
|---|---|---|
| faithfulness | 0.925 | 19/19 |
| answer_relevancy | 0.814 | 19/19 |
| llm_context_precision_without_reference | 0.875 | 2/19 |

## Pipeline mechanics

- modes: {'llm': 19, 'abstain': 0, 'extractive': 4}
- gold doc in retrieved context: 100.0%
- claims: 55 total, 2.39/answer, validator dropped 0, number-grounding flagged 4 (grounded rate 0.927)
- JSON parse retries (truncation guard): 0
- latency medians: retrieval 8793 ms · answer 37.79 s · end-to-end 50.5 s

## Judge agreement (faithfulness)

- second judge `kilo-free:poolside/laguna-s-2.1:free`: n=12 (different model served each judge), mean |Δ| = 0.042, Pearson r = 0.322

## Per query

| # | mode | faith. | relev. | ctx prec. | gold rank | claims | e2e s | query |
|---|---|---|---|---|---|---|---|---|
| 1 | llm | 1.0 | 0.8709 | 0.75 | 1 | 4 | 78.56 | periodic updation of KYC records for NBFC customers |
| 2 | llm | 1.0 | 0.7979 | 1.0 | 1 | 5 | 41.32 | periodic updation of KYC records for high risk customers of  |
| 3 | llm | 1.0 | 0.9182 | None | 1 | 1 | 66.72 | minimum capital adequacy ratio requirement for commercial ba |
| 4 | llm | 1.0 | 0.9601 | None | 1 | 5 | 64.7 | liquidity coverage ratio LCR requirement for banks |
| 5 | llm | 1.0 | 0.7813 | None | 1 | 1 | 43.18 | capital requirement for housing finance companies |
| 6 | llm | 0.8571 | 0.7966 | None | 1 | 4 | 53.06 | credit card issuance norms for banks |
| 7 | extractive | None | None | None | 1 | 0 | 59.9 | issuance and conduct of credit cards by NBFCs |
| 8 | llm | 0.75 | 0.7747 | None | 1 | 3 | 45.3 | asset classification norms 90 days overdue provisioning for  |
| 9 | llm | 1.0 | 0.7781 | None | 1 | 2 | 35.29 | treatment of wilful defaulters suit filing by banks |
| 10 | llm | 1.0 | 0.9246 | None | 1 | 2 | 23.35 | peer to peer lending platform leverage ratio for NBFC |
| 11 | llm | 1.0 | 0.8371 | None | 5 | 1 | 41.57 | microfinance loan eligibility household income limit |
| 12 | llm | 0.8 | 0.8515 | None | 5 | 3 | 34.44 | core investment companies registration and net asset require |
| 13 | llm | 0.4444 | 0.8477 | None | 4 | 4 | 50.5 | securitisation transactions minimum retention requirement |
| 14 | extractive | None | None | None | 1 | 0 | 53.0 | code of conduct for recovery agents and repossession |
| 15 | llm | 1.0 | 0.8853 | None | 1 | 2 | 39.86 | fit and proper criteria for directors of commercial banks |
| 16 | llm | 1.0 | 0.8164 | None | 1 | 4 | 29.54 | approval for setting up digital banking channels by banks |
| 17 | llm | 1.0 | 0.8552 | None | 1 | 2 | 30.03 | foreign exchange risk reserve requirement for authorised dea |
| 18 | extractive | None | None | None | 1 | 0 | 59.75 | branch authorisation for NBFC opening of new branches |
| 19 | llm | 0.7143 | 0.5984 | None | 4 | 2 | 53.77 | know your customer requirements for co-operative banks |
| 20 | llm | 1.0 | 0.8254 | None | 1 | 4 | 50.69 | income recognition and asset classification norms for region |
| 21 | extractive | None | None | None | 3 | 0 | 55.66 | compounding of contraventions under FEMA |
| 22 | llm | 1.0 | 0.773 | None | 1 | 4 | 53.59 | statutory audit requirements for commercial banks |
| 23 | llm | 1.0 | 0.5792 | None | 1 | 2 | 36.58 | prepaid payment instrument issuance and loading limits |
