# RegIntel answer evaluation — 2026-10-22T22-51-25

Requested pins: answer `groq:openai/gpt-oss-120b` · judge `gemini:gemini-3.5-flash-lite -> gemini:gemini-2.5-flash -> gemini:gemini-3.1-flash-lite -> gemini:gemini-2.5-flash-lite` · judge 2 `gemini:gemini-3.7-flash` · strict pin: False · context 1024 tok/total · corpus `v1-182406aedd` · k=6 · scored 89/100 answers (mode=llm).

**Served by** (pins are only requests; failover may serve other models):

- answers: {'gemini/gemini-3.5-flash-lite': 63, 'groq/openai/gpt-oss-120b': 25, 'gemini/gemini-2.5-flash': 9, 'gemini/gemini-3.8-flash': 1}
- judge (all metrics): {'gemini/gemini-3.5-flash-lite': 676, 'gemini/gemini-2.5-flash': 40, 'gemini/gemini-2.5-flash-lite': 21, 'gemini/gemini-3.1-flash-lite': 1}
- judge 2 (faithfulness): {}

## Answer-level metrics (RAGAS)

| metric | mean | judged |
|---|---|---|
| faithfulness | 0.626 | 89/89 |
| answer_relevancy | 0.801 | 89/89 |
| llm_context_precision_without_reference | 0.704 | 89/89 |

## Pipeline mechanics

- modes: {'llm': 89, 'abstain': 9, 'extractive': 2}
- gold doc in retrieved context: 85.0%
- claims: 298 total, 2.98/answer, validator dropped 15, number-grounding flagged 58 (grounded rate 0.805)
- JSON parse retries (truncation guard): 7
- latency medians: retrieval 12456 ms · answer 18.39 s · end-to-end 34.1 s

## Per query

| # | mode | faith. | relev. | ctx prec. | gold rank | claims | e2e s | query |
|---|---|---|---|---|---|---|---|---|
| 1 | llm | 1.0 | 0.8154 | 1.0 | 1 | 2 | 43.6 | periodic updation of KYC records for high-risk customers of  |
| 2 | llm | 0.0 | 0.8161 | 0.0 | 1 | 2 | 11.93 | interest rate offered on savings deposits by commercial bank |
| 3 | llm | 0.1429 | 0.7666 | 0.0 | 1 | 2 | 10.45 | benchmark rate and spread for interest rates on advances |
| 4 | llm | 0.875 | 0.9056 | 0.0 | 1 | 2 | 11.28 | cash reserve ratio and statutory liquidity ratio maintenance |
| 5 | llm | 0.6923 | 0.7442 | 0.4167 | 1 | 4 | 16.11 | classification and valuation of the investment portfolio inc |
| 6 | llm | 0.9 | 0.7654 | 0.5 | 5 | 3 | 38.22 | income recognition, asset classification and provisioning fo |
| 7 | llm | 0.6 | 0.7286 | 0.0 | 3 | 5 | 20.83 | resolution of stressed assets and restructuring of advances |
| 8 | llm | 1.0 | 0.9278 | 0.5 | 2 | 1 | 12.87 | minimum retention requirement for securitisation transaction |
| 9 | llm | 0.5 | 0.7721 | 0.25 | 1 | 5 | 146.79 | transfer and distribution of credit risk through credit deri |
| 10 | llm | 0.25 | 0.6725 | 0.0 | 1 | 5 | 17.79 | large exposure limits and concentration risk management |
| 11 | llm | 0.5455 | 0.8415 | 0.3333 | 3 | 5 | 16.64 | credit risk management framework and appraisal standards |
| 12 | extractive | None | None | None | 1 | 0 | 67.61 | credit facilities and loans and advances by commercial banks |
| 13 | llm | 0.5 | 0.8511 | 1.0 | 1 | 4 | 13.15 | capital adequacy ratios and Basel III capital requirements |
| 14 | llm | 0.5625 | 0.7757 | 1.0 | 1 | 5 | 17.14 | declaration of dividend and remittance of profits by commerc |
| 15 | llm | 0.7778 | 0.8407 | 1.0 | 2 | 4 | 43.16 | capital charge for credit risk under the standardised approa |
| 16 | llm | 0.5333 | 0.7764 | 0.8333 | 4 | 6 | 43.42 | minimum capital requirements for market risk |
| 17 | llm | 0.1818 | 0.8087 | 0.0 | 4 | 5 | 18.85 | CVA framework for derivative exposures of commercial banks |
| 18 | llm | 0.55 | 0.8565 | 1.0 | None | 5 | 14.62 | board governance and responsibilities of commercial banks |
| 19 | llm | 0.5263 | 0.7369 | 1.0 | 1 | 0 | 63.93 | acquisition and holding of shares or voting rights in bankin |
| 20 | llm | 0.7333 | 0.8217 | 1.0 | 2 | 6 | 23.38 | internal audit function and audit committee requirements |
| 21 | llm | 0.6429 | 0.7517 | 1.0 | 1 | 4 | 15.8 | statutory audit and appointment of auditors |
| 22 | llm | 0.75 | 0.7246 | 0.5833 | 1 | 5 | 14.59 | compliance function and chief compliance officer |
| 23 | extractive | None | None | None | 1 | 0 | 58.07 | fraud risk management and reporting |
| 24 | llm | 0.9231 | 0.7562 | 1.0 | 1 | 3 | 15.48 | concurrent audit coverage for commercial banks |
| 25 | llm | 1.0 | 0.828 | 1.0 | 1 | 6 | 11.91 | managing risks in outsourcing arrangements |
| 26 | llm | 0.7059 | 0.7044 | 0.5 | 1 | 5 | 20.26 | cybersecurity, technology risk and resilience |
| 27 | llm | 0.1579 | 0.8335 | 1.0 | 1 | 4 | 21.12 | digital payment security controls |
| 28 | llm | 0.1429 | 0.7715 | 0.0 | 1 | 8 | 107.7 | credit card and debit card issuance and conduct |
| 29 | llm | 0.6364 | 0.7835 | 0.8333 | 1 | 4 | 20.36 | authorisation of digital banking channels |
| 30 | llm | 0.5882 | 0.6874 | 0.8333 | 1 | 6 | 23.06 | branch authorisation and opening of new branches |
| 31 | llm | 0.6667 | 0.8579 | 1.0 | 1 | 5 | 31.23 | periodic updation of KYC records for NBFC customers |
| 32 | llm | 0.4583 | 0.923 | 1.0 | 1 | 5 | 33.01 | acceptance of public deposits by NBFCs |
| 33 | llm | 0.75 | 0.8706 | 0.5 | 1 | 4 | 39.42 | capital adequacy norms for NBFCs |
| 34 | llm | 0.75 | 0.8565 | 1.0 | None | 4 | 114.35 | income recognition and asset classification for NBFCs |
| 35 | llm | 1.0 | 0.716 | 1.0 | 1 | 6 | 41.86 | resolution of stressed assets for NBFCs |
| 36 | abstain | None | None | None | 1 | 0 | 56.22 | securitisation transactions and minimum retention for NBFCs |
| 37 | llm | 0.5714 | 0.8066 | 0.3333 | 2 | 3 | 24.31 | transfer and distribution of credit risk by NBFCs |
| 38 | llm | 0.2143 | 0.8649 | 1.0 | 1 | 4 | 43.35 | concentration risk and large exposure limits for NBFCs |
| 39 | abstain | None | None | None | 6 | 3 | 45.46 | credit risk management framework for NBFCs |
| 40 | abstain | None | None | None | None | 0 | 56.93 | credit facilities and loans by NBFCs |
| 41 | llm | 0.6667 | 0.8486 | 1.0 | 2 | 5 | 37.49 | classification and valuation of the NBFC investment portfoli |
| 42 | llm | 0.6 | 0.8367 | 1.0 | 1 | 3 | 121.62 | asset liability management for NBFCs |
| 43 | llm | 0.6 | 0.7861 | 0.5 | None | 7 | 38.85 | governance framework for NBFCs |
| 44 | llm | 0.4 | 0.8287 | 1.0 | 1 | 4 | 37.12 | internal audit function for NBFCs |
| 45 | llm | 0.9091 | 0.8996 | 1.0 | 1 | 5 | 27.6 | statutory audit requirements for NBFCs |
| 46 | llm | 0.6 | 0.8456 | 1.0 | 1 | 3 | 17.01 | compliance function for NBFCs |
| 47 | llm | 0.75 | 0.909 | 1.0 | 1 | 3 | 37.96 | fraud risk management for NBFCs |
| 48 | llm | 0.4444 | 0.8227 | 1.0 | 1 | 2 | 78.69 | outsourcing risk management for NBFCs |
| 49 | llm | 0.0 | 0.8288 | 0.0 | 2 | 2 | 41.29 | cybersecurity and technology risk for NBFCs |
| 50 | llm | 0.0 | 0.7535 | 0.0 | 1 | 3 | 23.0 | credit card issuance and conduct by NBFCs |
| 51 | llm | 0.3846 | 0.8872 | 1.0 | 1 | 4 | 71.33 | digital payment security controls for NBFCs |
| 52 | llm | 1.0 | 0.7277 | 1.0 | 1 | 5 | 34.32 | branch authorisation for NBFCs |
| 53 | llm | 0.9091 | 0.7102 | 0.4167 | 3 | 3 | 41.48 | NBFC registration, exemptions and regulatory framework |
| 54 | llm | 0.1111 | 0.7681 | 0.0 | 1 | 3 | 33.8 | microfinance institution directions for NBFCs |
| 55 | llm | 0.125 | 0.6619 | 0.0 | 1 | 3 | 34.34 | peer-to-peer lending platform directions |
| 56 | llm | 0.3333 | 0.8457 | 1.0 | 1 | 5 | 31.26 | account aggregator directions for NBFCs |
| 57 | llm | 0.0 | 0.793 | 1.0 | 4 | 2 | 27.41 | responsible business conduct for NBFCs |
| 58 | llm | 1.0 | 0.7657 | 0.8333 | 1 | 4 | 86.05 | internal ombudsman for NBFC customer complaints |
| 59 | llm | 0.3333 | 0.7206 | 0.5833 | None | 5 | 85.54 | credit information reporting by NBFCs |
| 60 | llm | 0.375 | 0.7948 | 1.0 | 2 | 3 | 42.47 | supervisory returns for NBFCs |
| 61 | abstain | None | None | None | None | 0 | 45.19 | commercial banks counterparty credit risk amendments effecti |
| 62 | llm | 1.0 | 0.7334 | 0.325 | 4 | 4 | 81.53 | commercial banks counterparty credit risk amendments effecti |
| 63 | abstain | None | None | None | None | 0 | 34.26 | commercial banks miscellaneous amendments effective April 20 |
| 64 | llm | 0.6667 | 0.7107 | 0.25 | 4 | 1 | 19.25 | commercial banks miscellaneous amendments effective April 20 |
| 65 | llm | 0.5 | 0.7653 | 1.0 | None | 1 | 31.39 | capital charge for credit risk under the standardised approa |
| 66 | llm | 0.5 | 0.7672 | 1.0 | 3 | 2 | 51.3 | capital charge for credit risk under the standardised approa |
| 67 | llm | 0.5 | 0.7829 | 1.0 | None | 3 | 28.75 | asset classification, provisioning and income recognition di |
| 68 | llm | 1.0 | 0.716 | 1.0 | 3 | 4 | 39.03 | asset classification, provisioning and income recognition di |
| 69 | llm | 1.0 | 0.6666 | 1.0 | None | 3 | 31.35 | minimum capital requirements for market risk |
| 70 | abstain | None | None | None | 4 | 0 | 17.96 | minimum capital requirements for market risk |
| 71 | llm | 0.8 | 0.7211 | 1.0 | None | 3 | 31.6 | CVA framework for derivative exposures of commercial banks |
| 72 | llm | 0.375 | 0.7334 | 1.0 | 4 | 4 | 40.9 | CVA framework for derivative exposures of commercial banks |
| 73 | llm | 0.6667 | 0.7959 | 0.5 | None | 2 | 83.69 | Kisan Credit Card scheme for commercial banks |
| 74 | llm | 0.1667 | 0.7796 | 0.0 | 1 | 3 | 36.69 | Kisan Credit Card scheme for commercial banks |
| 75 | abstain | None | None | None | None | 0 | 24.02 | responsible business conduct directions for commercial banks |
| 76 | abstain | None | None | None | 1 | 0 | 19.54 | responsible business conduct directions for commercial banks |
| 77 | llm | 0.5 | 0.8441 | 1.0 | None | 2 | 35.95 | financial statements presentation and disclosures for commer |
| 78 | llm | 0.2 | 0.8038 | 0.5833 | 1 | 3 | 27.07 | financial statements presentation and disclosures for commer |
| 79 | llm | 0.2 | 0.7746 | 1.0 | None | 3 | 20.6 | transfer and distribution of credit risk by NBFCs |
| 80 | llm | 0.625 | 0.7747 | 0.25 | 2 | 3 | 42.55 | transfer and distribution of credit risk by NBFCs |
| 81 | llm | 1.0 | 0.7973 | 1.0 | 1 | 3 | 70.34 | What does the Commercial Banks Forthcoming Instructions Amen |
| 82 | llm | 0.75 | 0.8091 | 0.0 | 1 | 2 | 32.55 | What change was made to the deferment of option premium in t |
| 83 | abstain | None | None | None | 1 | 0 | 11.89 | What legal opinion requirement for clearing member banks was |
| 84 | llm | 1.0 | 0.674 | 0.5 | 1 | 3 | 24.83 | How is incurred CVA loss treated for exposure at default aft |
| 85 | llm | 1.0 | 0.912 | 1.0 | 1 | 1 | 28.97 | When does the Commercial Banks Forthcoming Instructions Amen |
| 86 | llm | 0.3333 | 0.7727 | 0.0 | 1 | 2 | 28.5 | What limits for non-option derivative contracts were amended |
| 87 | llm | 1.0 | 0.9085 | 1.0 | 1 | 1 | 34.1 | What change did the Small Finance Banks Miscellaneous Amendm |
| 88 | llm | 1.0 | 0.8815 | 1.0 | 1 | 1 | 85.52 | Which directions does RBI/2026-27/284 amend? |
| 89 | llm | 0.6667 | 0.7934 | 1.0 | 1 | 1 | 30.94 | What is the daily minimum cash reserve maintenance requireme |
| 90 | llm | 0.75 | 0.8652 | 1.0 | 1 | 1 | 26.41 | What undertaking must an authorised dealer obtain before off |
| 91 | llm | 1.0 | 0.8929 | 1.0 | 1 | 4 | 33.22 | What are the Foreign Exchange Risk Reserve requirements for  |
| 92 | llm | 0.8 | 0.9317 | 0.0 | 1 | 3 | 94.14 | What changes did the October 2026 circulars make to Risk Man |
| 93 | llm | 1.0 | 0.6677 | 0.5 | 2 | 2 | 26.25 | What does the UAPA notification say about the UNSC 1267/1989 |
| 94 | llm | 1.0 | 0.928 | 1.0 | 1 | 1 | 12.34 | Which entities are addressed by the October 2026 UAPA notifi |
| 95 | llm | 0.6667 | 0.8062 | 1.0 | 1 | 1 | 37.3 | What is the reference number of the amendment to the Commerc |
| 96 | llm | 1.0 | 0.8858 | 1.0 | 1 | 1 | 53.64 | What is the reference number of the Small Finance Banks Misc |
| 97 | llm | 1.0 | 0.7444 | 1.0 | 1 | 1 | 129.48 | What paragraph of the Commercial Banks Capital Adequacy Dire |
| 98 | llm | 1.0 | 0.9138 | 1.0 | 1 | 2 | 85.2 | What paragraph of the Payments Banks Prudential Norms on Cap |
| 99 | llm | 1.0 | 0.838 | 1.0 | 1 | 1 | 55.95 | What is the effective date of the Kisan Credit Card Scheme D |
| 100 | llm | 0.6667 | 0.8993 | 1.0 | 1 | 2 | 37.2 | What is the effective date of the Commercial Banks Minimum C |
