# RegIntel answer evaluation — 2026-10-17T19-38-16

Answer model `groq:openai/gpt-oss-120b` · judge `groq:openai/gpt-oss-120b` · corpus `v1-53b832bb59` · k=6 · scored 19/23 answers (mode=llm).

## Answer-level metrics (RAGAS)

| metric | mean | judged |
|---|---|---|
| faithfulness | 0.935 | 19/19 |
| answer_relevancy | 0.797 | 19/19 |
| llm_context_precision_without_reference | 0.781 | 19/19 |

## Pipeline mechanics

- modes: {'llm': 19, 'abstain': 4, 'extractive': 0}
- gold doc in retrieved context: 100.0%
- claims: 65 total, 2.83/answer, validator dropped 0, number-grounding flagged 9 (grounded rate 0.862)
- JSON parse retries (truncation guard): 0
- latency medians: retrieval 9895 ms · answer 17.81 s · end-to-end 31.65 s

## Judge agreement (faithfulness)

- second judge `gemini:gemini-3.7-flash`: n=19, mean |Δ| = 0.001, Pearson r = 0.999

## Per query

| # | mode | faith. | relev. | ctx prec. | gold rank | claims | e2e s | query |
|---|---|---|---|---|---|---|---|---|
| 1 | llm | 1.0 | 0.8556 | 0.75 | 1 | 5 | 39.42 | periodic updation of KYC records for NBFC customers |
| 2 | llm | 1.0 | 0.8078 | 1.0 | 1 | 4 | 14.27 | periodic updation of KYC records for high risk customers of  |
| 3 | llm | 1.0 | 0.8835 | 0.8333 | 1 | 2 | 75.69 | minimum capital adequacy ratio requirement for commercial ba |
| 4 | llm | 1.0 | 0.8397 | 1.0 | 1 | 3 | 13.17 | liquidity coverage ratio LCR requirement for banks |
| 5 | llm | 0.8 | 0.7997 | 1.0 | 1 | 2 | 68.25 | capital requirement for housing finance companies |
| 6 | llm | 1.0 | 0.7027 | 1.0 | 1 | 4 | 28.79 | credit card issuance norms for banks |
| 7 | llm | 1.0 | 0.6927 | 0.5 | 1 | 4 | 24.69 | issuance and conduct of credit cards by NBFCs |
| 8 | abstain | None | None | None | 1 | 0 | 38.0 | asset classification norms 90 days overdue provisioning for  |
| 9 | abstain | None | None | None | 1 | 0 | 25.87 | treatment of wilful defaulters suit filing by banks |
| 10 | llm | 1.0 | 0.8733 | 0.5 | 1 | 2 | 25.85 | peer to peer lending platform leverage ratio for NBFC |
| 11 | llm | 0.8889 | 0.8561 | 1.0 | 5 | 3 | 46.68 | microfinance loan eligibility household income limit |
| 12 | abstain | None | None | None | 5 | 0 | 48.73 | core investment companies registration and net asset require |
| 13 | llm | 0.8333 | 0.7894 | 0.25 | 4 | 4 | 37.93 | securitisation transactions minimum retention requirement |
| 14 | llm | 0.6 | 0.6975 | 1.0 | 1 | 2 | 32.91 | code of conduct for recovery agents and repossession |
| 15 | llm | 1.0 | 0.8641 | 0.7 | 1 | 6 | 30.75 | fit and proper criteria for directors of commercial banks |
| 16 | llm | 1.0 | 0.7822 | 1.0 | 1 | 4 | 35.51 | approval for setting up digital banking channels by banks |
| 17 | llm | 1.0 | 0.8811 | 1.0 | 1 | 3 | 24.56 | foreign exchange risk reserve requirement for authorised dea |
| 18 | llm | 1.0 | 0.7775 | 1.0 | 1 | 4 | 36.1 | branch authorisation for NBFC opening of new branches |
| 19 | abstain | None | None | None | 4 | 0 | 29.76 | know your customer requirements for co-operative banks |
| 20 | llm | 1.0 | 0.7533 | 0.8333 | 1 | 3 | 26.24 | income recognition and asset classification norms for region |
| 21 | llm | 0.8 | 0.8127 | 0.4667 | 3 | 4 | 30.85 | compounding of contraventions under FEMA |
| 22 | llm | 1.0 | 0.8334 | 0.5 | 1 | 3 | 31.65 | statutory audit requirements for commercial banks |
| 23 | llm | 0.8333 | 0.645 | 0.5 | 1 | 3 | 32.09 | prepaid payment instrument issuance and loading limits |

_Judge models seen in cache during scoring: {'gemini-3.5-flash-lite': 164, 'openai/gpt-oss-120b': 7}_
