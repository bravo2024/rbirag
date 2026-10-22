#!/usr/bin/env python3
"""Build the draft v2 evaluation set (120 questions) + a human review sheet.

Categories
----------
* answerable  (60): natural questions over current Master Directions, one gold
  document each (some permit a broad + entity-specific pair).
* as_of       (20): ten documents x (before effective date, on/after it). The
  harness checks whether the probe document is retrieved or correctly excluded.
* amendment   (20): questions about the October 2026 notifications and the
  Directions they amend; gold includes the notification and its parent.
* unanswerable(20): out-of-regulator and in-domain traps; expected abstention.

Every row is ``review_status = "draft"``. The questions were authored from
document titles/sections and the notification texts, not from retrieval
results; gold documents are explicit. A domain reviewer must still check them
before the set is treated as gold — see evals/eval_set_v2_README.md.

    python scripts/build_eval_set_v2.py --force
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from regintel import store  # noqa: E402

OUT = ROOT / "evals" / "eval_set_v2.jsonl"
REVIEW = ROOT / "evals" / "eval_set_v2_review.csv"

# (institution, question, gold_doc_ids, expected_topic)
ANSWERABLE = [
    ("bank", "periodic updation of KYC records for high-risk customers of commercial banks", ["md13141"], "KYC periodic updation frequency and risk categories"),
    ("bank", "interest rate offered on savings deposits by commercial banks", ["md13157"], "savings deposit interest rate rules"),
    ("bank", "benchmark rate and spread for interest rates on advances", ["md13149"], "lending rate framework"),
    ("bank", "cash reserve ratio and statutory liquidity ratio maintenance requirements", ["md13160"], "CRR/SLR maintenance"),
    ("bank", "classification and valuation of the investment portfolio including held-to-maturity", ["md13148"], "investment classification and valuation"),
    ("bank", "income recognition, asset classification and provisioning for non-performing assets", ["md13146"], "IRAC norms"),
    ("bank", "resolution of stressed assets and restructuring of advances", ["md13145"], "stressed asset resolution"),
    ("bank", "minimum retention requirement for securitisation transactions", ["md13150"], "securitisation retention"),
    ("bank", "transfer and distribution of credit risk through credit derivatives", ["md13151"], "credit risk transfer"),
    ("bank", "large exposure limits and concentration risk management", ["md13152"], "large exposures"),
    ("bank", "credit risk management framework and appraisal standards", ["md13153"], "credit risk framework"),
    ("bank", "credit facilities and loans and advances by commercial banks", ["md13156"], "credit facilities"),
    ("bank", "capital adequacy ratios and Basel III capital requirements", ["md13159"], "capital adequacy"),
    ("bank", "declaration of dividend and remittance of profits by commercial banks", ["md13319"], "dividend rules"),
    ("bank", "capital charge for credit risk under the standardised approach for commercial banks", ["md13399"], "standardised approach capital charge"),
    ("bank", "minimum capital requirements for market risk", ["md13705"], "market risk capital"),
    ("bank", "CVA framework for derivative exposures of commercial banks", ["md13729"], "CVA framework"),
    ("bank", "board governance and responsibilities of commercial banks", ["md13161"], "governance"),
    ("bank", "acquisition and holding of shares or voting rights in banking companies", ["md13166"], "share acquisition limits"),
    ("bank", "internal audit function and audit committee requirements", ["md13640"], "internal audit"),
    ("bank", "statutory audit and appointment of auditors", ["md13639"], "statutory audit"),
    ("bank", "compliance function and chief compliance officer", ["md13645"], "compliance function"),
    ("bank", "fraud risk management and reporting", ["md13641"], "fraud risk"),
    ("bank", "concurrent audit coverage for commercial banks", ["md13644"], "concurrent audit"),
    ("bank", "managing risks in outsourcing arrangements", ["md13139"], "outsourcing risk"),
    ("bank", "cybersecurity, technology risk and resilience", ["md13643"], "cyber resilience"),
    ("bank", "digital payment security controls", ["md13642"], "digital payment security"),
    ("bank", "credit card and debit card issuance and conduct", ["md13155"], "card issuance"),
    ("bank", "authorisation of digital banking channels", ["md13163"], "digital channels"),
    ("bank", "branch authorisation and opening of new branches", ["md13164"], "branch authorisation"),
    ("nbfc", "periodic updation of KYC records for NBFC customers", ["md12943"], "KYC periodic updation"),
    ("nbfc", "acceptance of public deposits by NBFCs", ["md12958"], "public deposits"),
    ("nbfc", "capital adequacy norms for NBFCs", ["md12959"], "capital adequacy"),
    ("nbfc", "income recognition and asset classification for NBFCs", ["md12948"], "IRAC"),
    ("nbfc", "resolution of stressed assets for NBFCs", ["md12947"], "stressed assets"),
    ("nbfc", "securitisation transactions and minimum retention for NBFCs", ["md12951"], "securitisation"),
    ("nbfc", "transfer and distribution of credit risk by NBFCs", ["md12952"], "credit risk transfer"),
    ("nbfc", "concentration risk and large exposure limits for NBFCs", ["md12953"], "concentration risk"),
    ("nbfc", "credit risk management framework for NBFCs", ["md12954"], "credit risk"),
    ("nbfc", "credit facilities and loans by NBFCs", ["md12957"], "credit facilities"),
    ("nbfc", "classification and valuation of the NBFC investment portfolio", ["md12950"], "investment classification"),
    ("nbfc", "asset liability management for NBFCs", ["md12949"], "ALM"),
    ("nbfc", "governance framework for NBFCs", ["md12960"], "governance"),
    ("nbfc", "internal audit function for NBFCs", ["md13589"], "internal audit"),
    ("nbfc", "statutory audit requirements for NBFCs", ["md13588"], "statutory audit"),
    ("nbfc", "compliance function for NBFCs", ["md13593"], "compliance"),
    ("nbfc", "fraud risk management for NBFCs", ["md13590"], "fraud risk"),
    ("nbfc", "outsourcing risk management for NBFCs", ["md12941"], "outsourcing"),
    ("nbfc", "cybersecurity and technology risk for NBFCs", ["md13592"], "cyber risk"),
    ("nbfc", "credit card issuance and conduct by NBFCs", ["md12956"], "credit cards"),
    ("nbfc", "digital payment security controls for NBFCs", ["md13591"], "digital payment security"),
    ("nbfc", "branch authorisation for NBFCs", ["md12962"], "branch authorisation"),
    ("nbfc", "NBFC registration, exemptions and regulatory framework", ["md12965"], "registration"),
    ("nbfc", "microfinance institution directions for NBFCs", ["md12933"], "microfinance"),
    ("nbfc", "peer-to-peer lending platform directions", ["md12934"], "P2P lending"),
    ("nbfc", "account aggregator directions for NBFCs", ["md12936"], "account aggregator"),
    ("nbfc", "responsible business conduct for NBFCs", ["md12942"], "conduct"),
    ("nbfc", "internal ombudsman for NBFC customer complaints", ["md13274", "md12586"], "internal ombudsman"),
    ("nbfc", "credit information reporting by NBFCs", ["md12955"], "credit information"),
    ("nbfc", "supervisory returns for NBFCs", ["md13587"], "supervisory returns"),
]

# (question, institution, as_of, probe_doc_id, expect_present, note)
AS_OF = [
    ("commercial banks counterparty credit risk amendments effective April 2027", "bank", "2026-10-20", "nn13730", False, "effective 2027-04-01"),
    ("commercial banks counterparty credit risk amendments effective April 2027", "bank", "2027-04-01", "nn13730", True, "effective 2027-04-01"),
    ("commercial banks miscellaneous amendments effective April 2027", "bank", "2026-10-20", "nn13731", False, "effective 2027-04-01"),
    ("commercial banks miscellaneous amendments effective April 2027", "bank", "2027-04-01", "nn13731", True, "effective 2027-04-01"),
    ("capital charge for credit risk under the standardised approach", "bank", "2026-10-20", "md13399", False, "effective 2027-04-01"),
    ("capital charge for credit risk under the standardised approach", "bank", "2027-04-01", "md13399", True, "effective 2027-04-01"),
    ("asset classification, provisioning and income recognition directions", "bank", "2026-10-20", "md13400", False, "effective 2027-04-01"),
    ("asset classification, provisioning and income recognition directions", "bank", "2027-04-01", "md13400", True, "effective 2027-04-01"),
    ("minimum capital requirements for market risk", "bank", "2026-10-20", "md13705", False, "effective 2027-04-01"),
    ("minimum capital requirements for market risk", "bank", "2027-04-01", "md13705", True, "effective 2027-04-01"),
    ("CVA framework for derivative exposures of commercial banks", "bank", "2026-10-20", "md13729", False, "effective 2027-04-01"),
    ("CVA framework for derivative exposures of commercial banks", "bank", "2027-04-01", "md13729", True, "effective 2027-04-01"),
    ("Kisan Credit Card scheme for commercial banks", "bank", "2026-10-20", "md13522", False, "effective 2027-01-01"),
    ("Kisan Credit Card scheme for commercial banks", "bank", "2027-01-01", "md13522", True, "effective 2027-01-01"),
    ("responsible business conduct directions for commercial banks", "bank", "2026-03-31", "md13140", False, "effective 2026-04-01"),
    ("responsible business conduct directions for commercial banks", "bank", "2026-04-01", "md13140", True, "effective 2026-04-01"),
    ("financial statements presentation and disclosures for commercial banks", "bank", "2026-03-31", "md13143", False, "effective 2026-04-01"),
    ("financial statements presentation and disclosures for commercial banks", "bank", "2026-04-01", "md13143", True, "effective 2026-04-01"),
    ("transfer and distribution of credit risk by NBFCs", "nbfc", "2025-12-31", "md12952", False, "effective 2026-01-01"),
    ("transfer and distribution of credit risk by NBFCs", "nbfc", "2026-01-01", "md12952", True, "effective 2026-01-01"),
]

# (institution, question, gold_doc_ids, expected_topic)
AMENDMENT = [
    ("bank", "What does the Commercial Banks Forthcoming Instructions Amendment Directions, 2026 change about counterparty credit risk?", ["nn13730", "md13137"], "CCP/clearing member definitions and counterparty credit risk capital"),
    ("bank", "What change was made to the deferment of option premium in the Commercial Banks Miscellaneous Directions?", ["nn13731", "md13136"], "deferment of option premium"),
    ("bank", "What legal opinion requirement for clearing member banks was amended in the Commercial Banks Capital Adequacy Directions in October 2026?", ["nn13732", "md13159"], "legal opinion for clearing member banks"),
    ("bank", "How is incurred CVA loss treated for exposure at default after the October 2026 amendment?", ["nn13732", "md13159"], "incurred CVA loss and EAD"),
    ("bank", "When does the Commercial Banks Forthcoming Instructions Amendment Directions, 2026 come into force?", ["nn13730"], "effective 2027-04-01"),
    ("bank", "What limits for non-option derivative contracts were amended in the Commercial Banks Miscellaneous Directions?", ["nn13740", "md13136"], "limits for non-option derivative contracts"),
    (None, "What change did the Small Finance Banks Miscellaneous Amendment Directions, 2026 make to non-option derivative contract limits?", ["nn13741", "md13105"], "limits for non-option derivative contracts"),
    ("bank", "Which directions does RBI/2026-27/284 amend?", ["nn13730", "md13137"], "Commercial Banks Forthcoming Instructions Directions, 2025"),
    ("bank", "What is the daily minimum cash reserve maintenance requirement for scheduled banks after the October 2026 change?", ["nn13736"], "90% of required CRR on all days"),
    ("bank", "What undertaking must an authorised dealer obtain before offering an INR foreign exchange derivative?", ["nn13737"], "no other dealer has hedged the same underlying exposure"),
    ("bank", "What are the Foreign Exchange Risk Reserve requirements for authorised dealers?", ["nn13738"], "FX Risk Reserve"),
    ("bank", "What changes did the October 2026 circulars make to Risk Management and Inter-Bank Dealings?", ["nn13737", "nn13738", "nn13739"], "FX derivatives, FX Risk Reserve, inter-bank dealings"),
    ("bank", "What does the UAPA notification say about the UNSC 1267/1989 sanctions list amendment?", ["nn13742"], "amendment of 03 entries"),
    ("bank", "Which entities are addressed by the October 2026 UAPA notification?", ["nn13742"], "banks, NBFCs, ARCs and other regulated entities"),
    ("bank", "What is the reference number of the amendment to the Commercial Banks Prudential Norms on Capital Adequacy Directions?", ["nn13732"], "RBI/2026-27/286"),
    (None, "What is the reference number of the Small Finance Banks Miscellaneous Amendment Directions, 2026?", ["nn13741"], "RBI/2026-27/295"),
    ("bank", "What paragraph of the Commercial Banks Capital Adequacy Directions deals with legal opinion for clearing member banks?", ["nn13732", "md13159"], "paragraph 85(6)(i)"),
    (None, "What paragraph of the Payments Banks Prudential Norms on Capital Adequacy Directions was amended by RBI/2026-27/288?", ["nn13734", "md13097"], "paragraph 52(5)(i)"),
    ("bank", "What is the effective date of the Kisan Credit Card Scheme Directions for commercial banks, 2026?", ["md13522"], "2027-01-01"),
    ("bank", "What is the effective date of the Commercial Banks Minimum Capital Requirements for Market Risk Directions, 2026?", ["md13705"], "2027-04-01"),
]

# (question, reason)
UNANSWERABLE = [
    ("What are SEBI's disclosure requirements for listed companies?", "other regulator (SEBI)"),
    ("What is IRDAI's solvency ratio requirement for insurers?", "other regulator (IRDAI)"),
    ("What are the US Federal Reserve's capital rules for bank holding companies?", "other regulator (Fed)"),
    ("What is the FCA's consumer duty for UK financial firms?", "other regulator (FCA)"),
    ("What are ESMA's margining requirements for central counterparties?", "other regulator (ESMA)"),
    ("What is the HMRC reporting threshold for cash transactions?", "other regulator (HMRC)"),
    ("What are MAS Singapore's outsourcing requirements for banks?", "other regulator (MAS)"),
    ("What is the Basel Committee's latest liquidity coverage ratio calibration?", "international standard, not an RBI instrument"),
    ("What are the SEC's rules on climate risk disclosure?", "other regulator (SEC)"),
    ("What is the ECB's supervisory review process for significant institutions?", "other regulator (ECB)"),
    ("What is today's RBI repo rate and did the MPC change it?", "MPC decision / market rate, not a Master Direction"),
    ("Summarise the RBI's latest monetary policy statement.", "press release / policy statement"),
    ("What did the RBI announce in its October 2026 press release on developmental and regulatory policies?", "press release"),
    ("What is the current USD/INR reference rate published by RBI?", "market rate, not a Master Direction"),
    ("What is the corporate tax rate applicable to Indian banks?", "tax law, not an RBI Master Direction"),
    ("What is the GST rate on financial services in India?", "GST law, not an RBI Master Direction"),
    ("What are the bank holiday dates for October 2026 in Maharashtra?", "bank holiday list, not in corpus"),
    ("What penalty did RBI impose on a bank for KYC violations in 2026?", "enforcement action, not in corpus"),
    ("What is the RBI's framework for central bank digital currency pilots?", "press release / pilot, not a Master Direction"),
    ("What is the RBI's stance on cryptocurrency trading in India?", "press release / policy stance, not a Master Direction"),
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the draft v2 eval set")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing set (manual review edits are lost)")
    args = ap.parse_args(argv)
    if OUT.exists() and not args.force:
        print(f"{OUT.relative_to(ROOT)} already exists; pass --force to regenerate")
        return 1

    rows: list[dict] = []
    for i, (inst, query, gold, topic) in enumerate(ANSWERABLE, 1):
        rows.append({"id": f"v2-a{i:02d}", "category": "answerable", "query": query,
                     "institution": inst, "as_of": None, "gold_doc_ids": gold,
                     "expected_topic": topic, "expected": "answer",
                     "review_status": "draft", "reviewer_notes": ""})
    for i, (query, inst, as_of, probe, present, note) in enumerate(AS_OF, 1):
        rows.append({"id": f"v2-t{i:02d}", "category": "as_of", "query": query,
                     "institution": inst, "as_of": as_of, "gold_doc_ids": [probe],
                     "probe_doc_id": probe, "expect_present": present,
                     "expected_topic": note, "expected": "included" if present else "excluded",
                     "review_status": "draft", "reviewer_notes": ""})
    for i, (inst, query, gold, topic) in enumerate(AMENDMENT, 1):
        rows.append({"id": f"v2-m{i:02d}", "category": "amendment", "query": query,
                     "institution": inst, "as_of": None, "gold_doc_ids": gold,
                     "expected_topic": topic, "expected": "answer",
                     "review_status": "draft", "reviewer_notes": ""})
    for i, (query, reason) in enumerate(UNANSWERABLE, 1):
        rows.append({"id": f"v2-u{i:02d}", "category": "unanswerable", "query": query,
                     "institution": None, "as_of": None, "gold_doc_ids": [],
                     "expected_topic": reason, "expected": "abstain",
                     "review_status": "draft", "reviewer_notes": ""})

    # sanity: unique ids and queries
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate ids"
    queries = [r["query"] for r in rows]
    dupes = {q for q in queries if queries.count(q) > 1}
    if dupes:
        print(f"note: {len(dupes)} repeated query text(s) (as-of before/after pairs)")

    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                   encoding="utf-8")
    docs = store.load_documents()
    cols = ["id", "category", "query", "institution", "as_of", "gold_doc_ids",
            "gold_titles", "probe_doc_id", "expect_present", "expected",
            "expected_topic", "review_status", "reviewer_notes"]
    with REVIEW.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in cols}
            out["gold_doc_ids"] = ";".join(r.get("gold_doc_ids") or [])
            out["gold_titles"] = "; ".join(
                docs[d]["title"] for d in (r.get("gold_doc_ids") or []) if d in docs)
            w.writerow(out)

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["category"]] = counts.get(r["category"], 0) + 1
    print(f"wrote {OUT.relative_to(ROOT)} ({len(rows)} questions: {counts})")
    print(f"wrote {REVIEW.relative_to(ROOT)} (review sheet)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
