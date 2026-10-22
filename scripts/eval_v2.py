#!/usr/bin/env python3
"""Evaluate the v2 question set category by category.

* answerable / amendment: document-level recall@k, MRR@k, nDCG@10 against
  ``gold_doc_ids`` (the same metrics as scripts/eval_retrieval.py).
* as_of: does the probe document appear exactly when the requested date is on
  or after its effective date?  (A date-filter test, not a relevance metric.)
* unanswerable: retrieval-level abstention rate.  Scoring this requires rerank
  scores; with ``--no-rerank`` the rows are reported as not scored.

Rows are split by ``review_status``; only ``reviewed`` rows should be cited as
gold.  The generated set starts as ``draft``.

    python scripts/eval_v2.py --no-rerank
    python scripts/eval_v2.py --only unanswerable          # rerank on
    python scripts/eval_v2.py --only answerable --k 5      # rerank on
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:  # Windows consoles default to cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

from regintel import config, store, util  # noqa: E402
from regintel.eval_metrics import hit_at_k, mrr_at_k, ndcg_at_k  # noqa: E402
from regintel.index import Engine, QueryFilters  # noqa: E402

SET_FILE = ROOT / "evals" / "eval_set_v2.jsonl"
RESULTS_DIR = ROOT / "evals" / "results"
CATEGORIES = ("answerable", "as_of", "amendment", "unanswerable")


def _filters(case: dict) -> QueryFilters | None:
    if not (case.get("institution") or case.get("as_of")):
        return None
    ents = (config.INSTITUTIONS[case["institution"]]["entities"]
            if case.get("institution") else frozenset())
    return QueryFilters(entities=frozenset(ents), as_of=case.get("as_of"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="RegIntel v2 eval set")
    ap.add_argument("--file", default=str(SET_FILE))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--only", choices=CATEGORIES, action="append",
                    help="run one category (repeatable); default: all")
    ap.add_argument("--reviewed-only", action="store_true",
                    help="score only rows whose review_status is 'reviewed'")
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    cases = [json.loads(line) for line in Path(args.file).read_text(encoding="utf-8").splitlines()
             if line.strip()]
    if args.only:
        cases = [c for c in cases if c["category"] in args.only]
    if args.reviewed_only:
        cases = [c for c in cases if c.get("review_status") == "reviewed"]
    if not cases:
        print("no cases selected")
        return 1

    eng = Engine()
    if not eng.ready:
        print("index not built — run scripts/build_index.py first")
        return 1
    docs_meta = store.load_documents()

    rerank = not args.no_rerank
    k = max(args.k, 10)
    rows: list[dict] = []
    for i, case in enumerate(cases, 1):
        res = eng.retrieve(case["query"], filters=_filters(case), k=k, rerank=rerank)
        docs = [h.doc_id for h in res["hits"]]
        gold = set(case.get("gold_doc_ids") or [])
        row = {"id": case["id"], "category": case["category"], "query": case["query"],
               "institution": case.get("institution"), "as_of": case.get("as_of"),
               "review_status": case.get("review_status", "draft"),
               "got": docs[:10], "gold": sorted(gold),
               "top_score": (res["hits"][0].scores.get("rerank") if res["hits"] else None),
               "n_hits": len(res["hits"]), "retrieval_abstain": bool(res["abstain"]),
               "timings_ms": res["trace"].get("timings_ms")}

        if case["category"] in ("answerable", "amendment"):
            row.update({
                "hit@k": hit_at_k(docs, gold, args.k),
                "rr": round(mrr_at_k(docs, gold, args.k), 3),
                "ndcg@10": round(ndcg_at_k(docs, gold, 10), 3),
                "recall@10": hit_at_k(docs, gold, 10),
            })
            mark = "PASS" if row["hit@k"] else "FAIL"
            print(f"[{i}/{len(cases)}] {mark} {case['id']} rr={row['rr']:.2f} "
                  f"{case['query'][:64]}")
        elif case["category"] == "as_of":
            probe = case["probe_doc_id"]
            present = probe in docs
            as_of = case["as_of"]
            leaks = sorted({d for d in docs
                            if (docs_meta.get(d, {}).get("effective_from") or "") > as_of})
            row.update({"probe_doc_id": probe, "probe_present": present,
                        "expect_present": bool(case["expect_present"]),
                        "future_leaks": leaks,
                        "passed": present == bool(case["expect_present"]) and not leaks})
            mark = "PASS" if row["passed"] else "FAIL"
            print(f"[{i}/{len(cases)}] {mark} {case['id']} present={present} "
                  f"expected={case['expect_present']} leaks={len(leaks)} "
                  f"{case['query'][:56]}")
        else:  # unanswerable
            scored = rerank
            row.update({"scored": scored,
                        "passed": (bool(res["abstain"]) if scored else None)})
            mark = ("ABSTAIN" if row["passed"] else "ANSWERED") if scored else "UNSCORED"
            print(f"[{i}/{len(cases)}] {mark} {case['id']} top={row['top_score']} "
                  f"{case['query'][:64]}")
        rows.append(row)

    def _summary(subset: list[dict]) -> dict:
        out: dict = {}
        rel = [r for r in subset if r["category"] in ("answerable", "amendment")]
        if rel:
            n = len(rel)
            out["retrieval"] = {
                "n": n,
                f"recall@{args.k}": round(sum(r["hit@k"] for r in rel) / n, 3),
                f"mrr@{args.k}": round(sum(r["rr"] for r in rel) / n, 3),
                "recall@10": round(sum(r["recall@10"] for r in rel) / n, 3),
                "ndcg@10": round(sum(r["ndcg@10"] for r in rel) / n, 3),
            }
            by_inst: dict[str, list[dict]] = defaultdict(list)
            for r in rel:
                by_inst[r.get("institution") or "all"].append(r)
            out["retrieval_by_institution"] = {
                inst: {"n": len(v),
                       f"recall@{args.k}": round(sum(x["hit@k"] for x in v) / len(v), 3)}
                for inst, v in by_inst.items()}
        ao = [r for r in subset if r["category"] == "as_of"]
        if ao:
            out["as_of"] = {"n": len(ao),
                            "pass_rate": round(sum(r["passed"] for r in ao) / len(ao), 3),
                            "false_include": sum(1 for r in ao if not r["expect_present"] and r["probe_present"]),
                            "false_exclude": sum(1 for r in ao if r["expect_present"] and not r["probe_present"]),
                            "future_leaks": sum(1 for r in ao if r.get("future_leaks"))}
        un = [r for r in subset if r["category"] == "unanswerable"]
        if un:
            scored = [r for r in un if r["scored"]]
            out["unanswerable"] = {"n": len(un), "scored": len(scored),
                                   "abstain_rate": (round(sum(r["passed"] for r in scored) / len(scored), 3)
                                                    if scored else None)}
        return out

    summary = _summary(rows)
    reviewed = [r for r in rows if r["review_status"] == "reviewed"]
    summary_reviewed = _summary(reviewed) if reviewed else None

    print("\n" + "=" * 72)
    for key, val in summary.items():
        print(f"{key}: {val}")
    if summary_reviewed:
        print(f"reviewed subset ({len(reviewed)} rows): {summary_reviewed}")

    ts = util.iso_now().replace(":", "-")
    suffix = f"_{args.tag}" if args.tag else ""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / f"eval_v2_{ts}{suffix}.json"
    payload = {"ts": ts, "file": str(args.file), "k": args.k, "rerank": rerank,
               "only": args.only, "n": len(rows), "summary": summary,
               "summary_reviewed": summary_reviewed, "rows": rows}
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    md = [f"# Eval set v2 — {ts}", "",
          f"Set: `{Path(args.file).name}` · {len(rows)} questions · k={args.k} · "
          f"rerank={'on' if rerank else 'off'}", ""]
    md += ["## Summary", ""]
    for cat, s in summary.items():
        if cat == "retrieval_by_institution":
            md += ["**retrieval by institution**", "",
                   "| institution | n | recall@k |", "|---|---:|---:|"]
            for inst, vals in s.items():
                md.append(f"| {inst} | {vals['n']} | {vals[f'recall@{args.k}']} |")
            md.append("")
            continue
        md += [f"**{cat}**", "", "| metric | value |", "|---|---|"]
        for key, val in s.items():
            md.append(f"| {key} | {val} |")
        md.append("")
    if summary_reviewed:
        md += [f"Reviewed subset: {len(reviewed)} rows (metrics above are all rows).", ""]
    md += ["## Failures / reviewed flags", "",
           "| id | category | query | gold | got |", "|---|---|---|---|---|"]
    for r in rows:
        bad = (r.get("passed") is False or r.get("hit@k") == 0)
        if bad:
            got = list(dict.fromkeys(r["got"]))[:5]
            md.append(f"| {r['id']} | {r['category']} | {r['query'][:80]} | "
                      f"{', '.join(r['gold']) or '—'} | {', '.join(got) or '—'} |")
    out_md = RESULTS_DIR / f"eval_v2_{ts}{suffix}.md"
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"wrote {out_json.relative_to(ROOT)}")
    print(f"wrote {out_md.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
