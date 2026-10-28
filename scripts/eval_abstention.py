#!/usr/bin/env python3
"""Abstention eval: out-of-scope questions must abstain, in-scope ones must not.

Negatives: evals/queries_negative.jsonl (out-of-scope questions with no
answer in the corpus). Two levels are measured per question:

* retrieval-level: the hybrid engine flags abstain when the best rerank logit
  is below ABSTAIN_SCORE (or nothing passes the filters). By default the eval
  mirrors the app's default gate — a flagged query is refused without an LLM
  call (``answer.refuse``); ``--no-gate`` reproduces the caution-only path;
* answer-level: the LLM answer itself sets ``"abstain": true``.

A negative counts as correctly abstained when either level fires (the
interaction refuses to answer). With an answers_*.json from
scripts/eval_answers.py the same levels are summarised for the in-scope gold
set, giving the false-abstention rate.

    python scripts/eval_abstention.py [--answers evals/results/answers_<ts>.json]
    python scripts/eval_abstention.py --file evals/queries_negative_holdout.jsonl --tag holdout

The +2.0 threshold was calibrated on queries_negative.jsonl, so that set is
not an honest test of it; queries_negative_holdout.jsonl was authored without
looking at scores and is only ever run at the already-chosen threshold.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:  # Windows consoles default to cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

from regintel import answer as answer_mod  # noqa: E402
from regintel import config, util  # noqa: E402
from regintel.index import Engine, QueryFilters  # noqa: E402

NEG_FILE = ROOT / "evals" / "queries_negative.jsonl"
RESULTS_DIR = ROOT / "evals" / "results"


def _pin(s: str) -> tuple[str, str]:
    provider, _, model = s.partition(":")
    return (provider.strip(), model.strip())


def run_one(eng: Engine, query: str, filters, pin, gate: bool = True) -> dict:
    res = eng.retrieve(query, filters=filters, k=config.TOP_K)
    threshold = float(res["trace"].get("abstain_threshold", config.ABSTAIN_SCORE))
    refused = bool(res["abstain"] and res["hits"] and gate)
    if refused:
        a = answer_mod.refuse(query, res["hits"], threshold=threshold)
    else:
        a = answer_mod.answer(query, res["hits"],
                              low_confidence=bool(res["abstain"] and res["hits"]),
                              llm_pin=pin)
    return {
        "query": query,
        "retrieval_abstain": bool(res["abstain"]),
        "refused_at_retrieval": refused,
        "llm_abstain": bool(a["abstain"]),
        "mode": a["mode"],
        "top_rerank": (res["hits"][0].scores.get("rerank") if res["hits"] else None),
        "n_hits": len(res["hits"]),
        "pipeline_abstained": bool(res["abstain"] or a["abstain"]),
        "answer_model": a["model"],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="RBIRAG abstention eval")
    ap.add_argument("--answer-pin", default="groq:openai/gpt-oss-120b")
    ap.add_argument("--answers", default=None, help="answers_*.json for the in-scope set")
    ap.add_argument("--no-gate", action="store_true",
                    help="do not emulate the app's refuse-without-LLM gate")
    ap.add_argument("--file", default=str(NEG_FILE), help="negatives JSON-lines file")
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    pin = _pin(args.answer_pin)
    gate = not args.no_gate
    neg_file = Path(args.file)
    cases = [json.loads(l) for l in neg_file.read_text(encoding="utf-8").splitlines()
             if l.strip()]
    eng = Engine()
    if not eng.ready:
        print("index not built — run scripts/sync.py first")
        return 1

    print(f"== negatives: {len(cases)} out-of-scope questions from {neg_file.name} "
          f"(pin {args.answer_pin}) ==")
    rows = []
    for i, case in enumerate(cases, 1):
        filt = None
        if case.get("institution"):
            filt = QueryFilters(entities=config.INSTITUTIONS[case["institution"]]["entities"])
        row = run_one(eng, case["query"], filt, pin, gate=gate)
        row["note"] = case.get("note", "")
        rows.append(row)
        print(f"[{i}/{len(cases)}] {'ABSTAIN' if row['pipeline_abstained'] else 'ANSWERED':8s} "
              f"retr={row['retrieval_abstain']} llm={row['llm_abstain']} "
              f"top={row['top_rerank']}  {case['query'][:58]}")

    n = len(rows)
    neg = {
        "n": n,
        "pipeline_abstain_rate": round(sum(r["pipeline_abstained"] for r in rows) / n, 3),
        "retrieval_abstain_rate": round(sum(r["retrieval_abstain"] for r in rows) / n, 3),
        "refused_at_retrieval": sum(r["refused_at_retrieval"] for r in rows),
        "llm_abstain_rate": round(sum(r["llm_abstain"] for r in rows) / n, 3),
        "extractive_fallbacks": sum(1 for r in rows if r["mode"] == "extractive"),
    }

    # in-scope side: false-abstention rate from a saved answer run
    pos = None
    if args.answers:
        blob = json.loads(Path(args.answers).read_text(encoding="utf-8"))
        recs = blob["records"]
        m = len(recs)
        pos = {
            "n": m,
            "pipeline_abstain_rate": round(
                sum(1 for r in recs if r["retrieval_abstain"] or r["abstain"]) / m, 3),
            "retrieval_abstain_rate": round(
                sum(1 for r in recs if r["retrieval_abstain"]) / m, 3),
            "llm_abstain_rate": round(sum(1 for r in recs if r["abstain"]) / m, 3),
        }

    ts = util.iso_now().replace(":", "-")
    suffix = f"_{args.tag}" if args.tag else ""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"abstention_{ts}{suffix}.json"
    out.write_text(json.dumps({
        "ts": ts, "answer_pin": args.answer_pin, "abstain_score": config.ABSTAIN_SCORE,
        "gate": gate, "negatives_file": str(neg_file), "negatives": neg, "negative_rows": rows, "in_scope": pos,
        "in_scope_source": args.answers}, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nnegatives: pipeline abstain {neg['pipeline_abstain_rate']:.1%} "
          f"(retrieval {neg['retrieval_abstain_rate']:.1%}, llm {neg['llm_abstain_rate']:.1%}, "
          f"extractive fallbacks {neg['extractive_fallbacks']})")
    if pos:
        print(f"in-scope (n={pos['n']}): pipeline false-abstain {pos['pipeline_abstain_rate']:.1%} "
              f"(retrieval {pos['retrieval_abstain_rate']:.1%}, llm {pos['llm_abstain_rate']:.1%})")
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
