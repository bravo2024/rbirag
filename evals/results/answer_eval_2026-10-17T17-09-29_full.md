# RegIntel answer evaluation — 2026-10-17T17-09-29

Answer model `groq:openai/gpt-oss-120b` · judge `groq:openai/gpt-oss-120b` · corpus `v1-53b832bb59` · k=6 · scored 20/23 answers (mode=llm).

## Answer-level metrics (RAGAS)

| metric | mean | judged |
|---|---|---|
| faithfulness | 0.945 | 20/20 |
| answer_relevancy | 0.8 | 20/20 |
| llm_context_precision_without_reference | 0.74 | 20/20 |

## Pipeline mechanics

- modes: {'llm': 20, 'abstain': 2, 'extractive': 1}
- gold doc in retrieved context: 100.0%
- claims: 70 total, 3.04/answer, validator dropped 0, number-grounding flagged 8 (grounded rate 0.886)
- JSON parse retries (truncation guard): 0
- latency medians: retrieval 8814 ms · answer 10.45 s · end-to-end 20.39 s

## Judge agreement (faithfulness)

- second judge `gemini:gemini-3.7-flash`: n=20, mean |Δ| = 0.022, Pearson r = 0.715

## Per query

| # | mode | faith. | relev. | ctx prec. | gold rank | claims | e2e s | query |
|---|---|---|---|---|---|---|---|---|
| 1 | llm | 0.9474 | 0.8493 | 0.3333 | 1 | 5 | 25.71 | periodic updation of KYC records for NBFC customers |
| 2 | llm | 1.0 | 0.8851 | 1.0 | 1 | 4 | 8.52 | periodic updation of KYC records for high risk customers of  |
| 3 | llm | 1.0 | 0.8073 | 0.8333 | 1 | 2 | 9.63 | minimum capital adequacy ratio requirement for commercial ba |
| 4 | llm | 1.0 | 0.8895 | 1.0 | 1 | 2 | 9.42 | liquidity coverage ratio LCR requirement for banks |
| 5 | llm | 1.0 | 0.7752 | 1.0 | 1 | 3 | 27.9 | capital requirement for housing finance companies |
| 6 | llm | 1.0 | 0.74 | 0.75 | 1 | 4 | 19.53 | credit card issuance norms for banks |
| 7 | llm | 1.0 | 0.7853 | 0.5 | 1 | 4 | 21.78 | issuance and conduct of credit cards by NBFCs |
| 8 | extractive | None | None | None | 1 | 0 | 29.7 | asset classification norms 90 days overdue provisioning for  |
| 9 | abstain | None | None | None | 1 | 0 | 14.62 | treatment of wilful defaulters suit filing by banks |
| 10 | llm | 0.75 | 0.8522 | 0.5 | 1 | 3 | 12.02 | peer to peer lending platform leverage ratio for NBFC |
| 11 | llm | 1.0 | 0.7416 | 1.0 | 4 | 3 | 13.97 | microfinance loan eligibility household income limit |
| 12 | llm | 0.75 | 0.8077 | 0.2 | 5 | 3 | 24.21 | core investment companies registration and net asset require |
| 13 | llm | 1.0 | 0.8377 | 0.2 | 5 | 4 | 25.27 | securitisation transactions minimum retention requirement |
| 14 | llm | 1.0 | 0.6974 | 1.0 | 1 | 4 | 21.77 | code of conduct for recovery agents and repossession |
| 15 | llm | 0.8667 | 0.815 | 0.7 | 1 | 5 | 27.78 | fit and proper criteria for directors of commercial banks |
| 16 | llm | 1.0 | 0.7657 | 1.0 | 1 | 4 | 20.39 | approval for setting up digital banking channels by banks |
| 17 | llm | 1.0 | 0.7699 | 1.0 | 1 | 4 | 17.0 | foreign exchange risk reserve requirement for authorised dea |
| 18 | llm | 1.0 | 0.7432 | 1.0 | 1 | 4 | 26.12 | branch authorisation for NBFC opening of new branches |
| 19 | llm | 0.7 | 0.6735 | 1.0 | 1 | 3 | 7.84 | know your customer requirements for co-operative banks |
| 20 | llm | 1.0 | 0.9364 | 0.8333 | 1 | 2 | 6.82 | income recognition and asset classification norms for region |
| 21 | llm | 0.8889 | 0.832 | 0.45 | 3 | 3 | 26.57 | compounding of contraventions under FEMA |
| 22 | llm | 1.0 | 0.7888 | 0.5 | 1 | 4 | 34.07 | statutory audit requirements for commercial banks |
| 23 | abstain | None | None | None | 1 | 0 | 20.1 | prepaid payment instrument issuance and loading limits |

_Judge models seen in cache during scoring: {'gemini-3.5-flash-lite': 135, 'gemini-3.6-flash': 16, 'gemini-3.1-flash-lite': 16, 'openai/gpt-oss-120b': 6, 'nvidia/nemotron-3-super-120b-a12b': 2}_
