# RBIRAG answer evaluation — 2026-10-28T21-41-55

Requested pins: answer `groq:openai/gpt-oss-120b` · judge `groq:openai/gpt-oss-120b` · judge 2 `gemini:gemini-3.7-flash` · strict pin: False · full context · corpus `v1-182406aedd` · k=6 · scored 83/100 answers (mode=llm).

**Served by** (pins are only requests; failover may serve other models):

- answers: {'groq/openai/gpt-oss-120b': 54, 'gemini/gemini-2.5-flash': 15, 'gemini/gemini-3.6-flash': 14, 'gemini/gemini-2.5-flash-lite': 10, 'gemini/gemini-3.5-flash-lite': 7}
- judge (all metrics): {'gemini/gemini-3.5-flash-lite': 492, 'gemini/gemini-3.1-flash-lite': 370, 'groq/openai/gpt-oss-120b': 31, 'gemini/gemini-2.5-flash-lite': 12, 'nvidia-build/nvidia/nemotron-3-ultra-550b-a55b': 8, 'kilo-free/poolside/laguna-s-2.1:free': 2, 'gemini/gemini-3.5-flash': 1, 'gemini/gemini-2.5-flash': 1}
- judge 2 (faithfulness): {}

## Answer-level metrics (RAGAS)

| metric | mean | judged |
|---|---|---|
| faithfulness | 0.913 | 80/83 |
| answer_relevancy | 0.803 | 83/83 |
| llm_context_precision_without_reference | 0.544 | 83/83 |

## Pipeline mechanics

- modes: {'llm': 83, 'abstain': 17, 'extractive': 0}
- gold doc in retrieved context: 95.0%
- claims: 317 total, 3.17/answer, validator dropped 3, number-grounding flagged 31 (grounded rate 0.902)
- JSON parse retries (truncation guard): 7
- latency medians: retrieval 14901 ms · answer 7.98 s · end-to-end 27.8 s

## Per query

| # | mode | faith. | relev. | ctx prec. | gold rank | claims | e2e s | query |
|---|---|---|---|---|---|---|---|---|
| 1 | llm | None | 0.8539 | 0.4167 | 1 | 6 | 54.22 | periodic updation of KYC records for NBFC customers |
| 2 | llm | 1.0 | 0.7943 | 1.0 | 1 | 4 | 25.77 | periodic updation of KYC records for high risk customers of  |
| 3 | llm | None | 0.8441 | 1.0 | 1 | 2 | 24.31 | minimum capital adequacy ratio requirement for commercial ba |
| 4 | llm | 1.0 | 0.867 | 1.0 | 1 | 2 | 30.17 | liquidity coverage ratio LCR requirement for banks |
| 5 | llm | 0.8333 | 0.8594 | 1.0 | 1 | 3 | 22.01 | capital requirement for housing finance companies |
| 6 | llm | 0.8182 | 0.7789 | 1.0 | 1 | 5 | 28.06 | credit card issuance norms for banks |
| 7 | llm | 1.0 | 0.7882 | 0.5 | 1 | 4 | 199.05 | issuance and conduct of credit cards by NBFCs |
| 8 | llm | 0.8333 | 0.731 | 0.8333 | 1 | 4 | 24.44 | asset classification norms 90 days overdue provisioning for  |
| 9 | llm | 1.0 | 0.7663 | 0.5 | 1 | 4 | 23.99 | treatment of wilful defaulters suit filing by banks |
| 10 | llm | 1.0 | 0.8639 | 0.5 | 1 | 2 | 25.81 | peer to peer lending platform leverage ratio for NBFC |
| 11 | llm | 0.5 | 0.8249 | 1.0 | 5 | 1 | 19.48 | microfinance loan eligibility household income limit |
| 12 | llm | 1.0 | 0.8696 | 0.2667 | 5 | 4 | 23.19 | core investment companies registration and net asset require |
| 13 | llm | 0.2 | 0.726 | 0.25 | 4 | 3 | 34.06 | securitisation transactions minimum retention requirement |
| 14 | llm | 1.0 | 0.6296 | 0.5 | 1 | 4 | 46.62 | code of conduct for recovery agents and repossession |
| 15 | llm | 1.0 | 0.77 | 0.7 | 1 | 3 | 16.68 | fit and proper criteria for directors of commercial banks |
| 16 | llm | 1.0 | 0.7686 | 1.0 | 1 | 4 | 28.78 | approval for setting up digital banking channels by banks |
| 17 | llm | 1.0 | 0.8401 | 1.0 | 1 | 1 | 20.19 | foreign exchange risk reserve requirement for authorised dea |
| 18 | llm | 1.0 | 0.7634 | 1.0 | 1 | 2 | 20.49 | branch authorisation for NBFC opening of new branches |
| 19 | abstain | None | None | None | 4 | 0 | 19.19 | know your customer requirements for co-operative banks |
| 20 | llm | 1.0 | 0.8249 | 0.8333 | 1 | 4 | 57.7 | income recognition and asset classification norms for region |
| 21 | llm | 0.7778 | 0.7509 | 0.0 | 3 | 4 | 31.27 | compounding of contraventions under FEMA |
| 22 | llm | 1.0 | 0.7323 | 0.5 | 1 | 5 | 17.43 | statutory audit requirements for commercial banks |
| 23 | llm | 1.0 | 0.65 | 0.0 | 1 | 3 | 22.87 | prepaid payment instrument issuance and loading limits |
| 24 | llm | 0.75 | 0.9563 | 1.0 | 3 | 1 | 20.04 | minimum capital adequacy ratio for small finance banks |
| 25 | abstain | None | None | None | 6 | 0 | 21.14 | capital adequacy ratio requirement for payments banks |
| 26 | llm | 1.0 | 0.8452 | 1.0 | 1 | 5 | 28.59 | kisan credit card scheme for small finance bank borrowers |
| 27 | llm | 0.9231 | 0.7399 | 0.5 | 1 | 3 | 56.93 | managing risks in outsourcing for local area banks |
| 28 | abstain | None | None | None | 1 | 0 | 101.64 | interest rate on deposits for urban co-operative banks |
| 29 | llm | 1.0 | 0.8042 | 0.3333 | 1 | 4 | 13.82 | fraud risk management framework for urban co-operative banks |
| 30 | llm | 0.9231 | 0.919 | 0.0 | 1 | 4 | 15.13 | fraud monitoring and reporting requirements for commercial b |
| 31 | llm | 0.5 | 0.8257 | 1.0 | 1 | 2 | 112.48 | cybersecurity and technology resilience framework for small  |
| 32 | llm | 1.0 | 0.8329 | 0.4167 | 1 | 4 | 20.96 | digital payment security controls for commercial banks |
| 33 | llm | 0.8571 | 0.8655 | 0.0 | 1 | 3 | 26.33 | cyber resilience for non-bank prepaid payment instrument iss |
| 34 | llm | 1.0 | 0.7985 | 0.5 | 1 | 4 | 20.54 | interest rate on deposits for small finance banks |
| 35 | llm | 0.8 | 0.8609 | 1.0 | 1 | 1 | 17.45 | interest rate on deposits for payments banks |
| 36 | llm | 1.0 | 0.8042 | 0.3333 | 1 | 4 | 19.83 | internal ombudsman mechanism for commercial banks |
| 37 | abstain | None | None | None | 2 | 2 | 31.68 | priority sector lending targets for small finance banks |
| 38 | llm | 1.0 | 0.6722 | 0.0 | 1 | 4 | 32.31 | know your customer requirements for asset reconstruction com |
| 39 | abstain | None | None | None | 1 | 0 | 28.97 | treatment of wilful defaulters by asset reconstruction compa |
| 40 | llm | 0.8333 | 0.7591 | 0.0 | 1 | 3 | 14.55 | internal ombudsman for credit information companies |
| 41 | llm | 1.0 | 0.763 | 0.5 | 2 | 2 | 17.48 | credit information reporting by non-banking financial compan |
| 42 | llm | 1.0 | 0.8244 | 1.0 | 1 | 4 | 18.13 | credit information reporting by commercial banks |
| 43 | llm | 0.625 | 0.7124 | 0.3333 | 2 | 5 | 60.29 | dividend declaration norms for non-banking financial compani |
| 44 | abstain | None | None | None | 1 | 1 | 20.2 | dividend declaration norms for small finance banks |
| 45 | abstain | None | None | None | 1 | 1 | 19.74 | debit card issuance and conduct for payments banks |
| 46 | llm | 0.6667 | 0.889 | 0.5 | 1 | 4 | 18.08 | margining requirements for non-centrally cleared OTC derivat |
| 47 | llm | 1.0 | 0.6547 | 1.0 | 3 | 2 | 22.74 | capital adequacy norms for all India financial institutions |
| 48 | llm | None | 0.7866 | 1.0 | 1 | 4 | 18.71 | managing risks in outsourcing for all India financial instit |
| 49 | llm | 0.8333 | 0.6892 | 0.5 | 1 | 4 | 41.95 | credit information reporting by all India financial institut |
| 50 | llm | 1.0 | 0.8211 | 1.0 | 1 | 3 | 37.08 | fraud risk management for all India financial institutions |
| 51 | llm | 1.0 | 0.7729 | 1.0 | 1 | 3 | 13.92 | asset liability management for all India financial instituti |
| 52 | llm | 0.9091 | 0.8219 | 0.0 | 3 | 4 | 15.36 | resolution of stressed assets for all India financial instit |
| 53 | abstain | None | None | None | 1 | 0 | 16.16 | capital adequacy norms for regional rural banks |
| 54 | llm | 1.0 | 0.8764 | 1.0 | 1 | 6 | 76.49 | cash reserve ratio and statutory liquidity ratio for regiona |
| 55 | llm | 0.75 | 0.8274 | 0.75 | 2 | 2 | 48.19 | treatment of wilful defaulters by regional rural banks |
| 56 | llm | 1.0 | 0.8098 | 0.5 | 1 | 4 | 72.6 | concentration risk management for regional rural banks |
| 57 | llm | 1.0 | 0.9123 | 1.0 | 1 | 1 | 23.22 | asset liability management for regional rural banks |
| 58 | llm | 0.6667 | 0.8142 | 0.0 | 1 | 4 | 22.39 | credit facilities for regional rural banks |
| 59 | llm | 1.0 | 0.8608 | 0.6389 | 1 | 2 | 59.92 | capital adequacy norms for local area banks |
| 60 | abstain | None | None | None | 1 | 1 | 26.64 | cash reserve ratio and statutory liquidity ratio for local a |
| 61 | llm | 1.0 | 0.7199 | 0.3333 | 4 | 5 | 26.36 | responsible business conduct for local area banks |
| 62 | llm | 0.75 | 0.8233 | 0.0 | 2 | 3 | 23.01 | treatment of wilful defaulters by local area banks |
| 63 | llm | 1.0 | 0.7744 | 1.0 | None | 5 | 19.09 | income recognition and asset classification norms for local  |
| 64 | llm | 0.7273 | 0.7695 | 0.0 | 1 | 4 | 22.58 | supervisory returns for local area banks |
| 65 | abstain | None | None | None | 1 | 3 | 136.53 | capital adequacy norms for urban co-operative banks |
| 66 | llm | 1.0 | 0.8593 | 1.0 | 2 | 2 | 16.42 | income recognition and asset classification norms for urban  |
| 67 | llm | 1.0 | 0.9409 | 0.5 | 1 | 2 | 17.57 | interest rates on advances for urban co-operative banks |
| 68 | llm | 1.0 | 0.7725 | 0.0 | 1 | 2 | 16.15 | compliance function for urban co-operative banks |
| 69 | llm | 1.0 | 0.8218 | 0.2 | 1 | 5 | 20.85 | concurrent audit for urban co-operative banks |
| 70 | llm | 1.0 | 0.8409 | 1.0 | 1 | 8 | 61.81 | licensing and regulatory classification for urban co-operati |
| 71 | abstain | None | None | None | 1 | 4 | 34.18 | capital adequacy norms for rural co-operative banks |
| 72 | abstain | None | None | None | 1 | 0 | 28.88 | cash reserve ratio and statutory liquidity ratio for rural c |
| 73 | abstain | None | None | None | 1 | 0 | 25.54 | interest rate on deposits for rural co-operative banks |
| 74 | llm | 1.0 | 0.7893 | 0.8333 | 1 | 0 | 33.38 | fraud risk management for rural co-operative banks |
| 75 | llm | 1.0 | 0.7982 | 1.0 | 1 | 5 | 37.65 | asset liability management for rural co-operative banks |
| 76 | llm | 0.7273 | 0.838 | 1.0 | 1 | 6 | 98.75 | managing risks in outsourcing for rural co-operative banks |
| 77 | llm | 0.6667 | 0.7246 | 1.0 | 1 | 2 | 55.47 | cash reserve ratio and statutory liquidity ratio for payment |
| 78 | llm | 0.8333 | 0.8158 | 0.0 | 3 | 2 | 51.13 | fraud risk management for payments banks |
| 79 | llm | 0.8571 | 0.9255 | 0.0 | 1 | 4 | 27.8 | supervisory returns for payments banks |
| 80 | abstain | None | None | None | 2 | 0 | 46.71 | branch authorisation for payments banks |
| 81 | llm | 0.8889 | 0.686 | 1.0 | None | 4 | 80.34 | responsible business conduct for payments banks |
| 82 | abstain | None | None | None | 1 | 3 | 88.56 | cash reserve ratio and statutory liquidity ratio for small f |
| 83 | llm | 1.0 | 0.8967 | 0.0 | None | 4 | 61.16 | credit facilities for small finance banks |
| 84 | llm | 0.9333 | 0.775 | 0.0 | 2 | 7 | 61.56 | branch authorisation for small finance banks |
| 85 | abstain | None | None | None | 1 | 0 | 34.64 | compliance function for small finance banks |
| 86 | abstain | None | None | None | 1 | 0 | 10.43 | internal audit function for small finance banks |
| 87 | llm | 0.625 | 0.7571 | 0.0 | 1 | 5 | 44.06 | securitisation transactions for small finance banks |
| 88 | llm | 1.0 | 0.9288 | 0.0 | 2 | 5 | 33.49 | interest rates on advances for commercial banks |
| 89 | llm | 1.0 | 0.8103 | 0.0 | 3 | 4 | 42.66 | concentration risk management for commercial banks |
| 90 | llm | 1.0 | 0.8554 | 0.0 | 1 | 3 | 31.2 | interest rate on deposits for commercial banks |
| 91 | llm | 1.0 | 0.863 | 0.75 | 1 | 6 | 81.8 | concurrent audit for commercial banks |
| 92 | llm | 1.0 | 0.9058 | 1.0 | 1 | 2 | 31.7 | climate finance and climate change risk for commercial banks |
| 93 | llm | 1.0 | 0.8627 | 0.0 | 1 | 4 | 17.55 | supervisory returns for commercial banks |
| 94 | llm | 1.0 | 0.7573 | 0.2 | 1 | 5 | 27.7 | credit risk management for non-banking financial companies |
| 95 | llm | 1.0 | 0.8282 | 0.5 | None | 6 | 48.47 | classification and valuation of investment portfolio for non |
| 96 | llm | 1.0 | 0.7942 | 0.0 | 1 | 4 | 40.49 | resolution of stressed assets for non-banking financial comp |
| 97 | llm | 1.0 | 0.7687 | 1.0 | 1 | 3 | 50.47 | cybersecurity framework for non-banking financial companies |
| 98 | llm | 1.0 | 0.7639 | 0.25 | None | 8 | 45.03 | acceptance of public deposits by non-banking financial compa |
| 99 | llm | 1.0 | 0.7419 | 1.0 | 1 | 4 | 39.52 | account aggregator framework for non-banking financial compa |
| 100 | llm | 1.0 | 0.6594 | 0.0 | 1 | 3 | 19.67 | cybersecurity framework for credit information companies |
