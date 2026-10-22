#!/usr/bin/env python3
"""Retrieval eval: gold-document recall@k, MRR@k and nDCG@10 over evals/queries.jsonl.

Each line: {"query": str, "gold_doc_ids": [str, ...], "institution": "bank|nbfc|null",
            "note": str}. A query counts as hit@k if any gold document appears among
the returned top-k passages; nDCG@10 and recall@10 use the top-10 ranking with
binary document-level gains.

    python scripts/eval_retrieval.py [--k 5] [--no-rerank]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:  # Windows consoles default to cp1252; results contain unicode titles
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

from regintel import config, util  # noqa: E402
from regintel.eval_metrics import hit_at_k, mrr_at_k, ndcg_at_k  # noqa: E402
from regintel.index import Engine, QueryFilters  # noqa: E402

EVAL_FILE = ROOT / "evals" / "queries.jsonl"
RESULTS_DIR = ROOT / "evals" / "results"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="RegIntel retrieval eval")
    ap.add_argument("--k", type=int, default=5,
                    help="recall/MRR cutoff (nDCG and recall@10 always use top-10)")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--file", default=str(EVAL_FILE), help="query set jsonl")
    ap.add_argument("--tag", default="", help="suffix for the results file")
    args = ap.parse_args(argv)

    cases = [json.loads(l) for l in Path(args.file).read_text(encoding="utf-8").splitlines()
             if l.strip()]
    eng = Engine()
    if not eng.ready:
        print("index not built — run scripts/build_index.py first")
        return 1

    retrieve_k = max(args.k, 10)
    rows, hit_total, mrr_total, ndcg_total, recall10_total = [], 0, 0.0, 0.0, 0
    for case in cases:
        filt = None
        if case.get("institution"):
            filt = QueryFilters(entities=config.INSTITUTIONS[case["institution"]]["entities"])
        res = eng.retrieve(case["query"], filters=filt, k=retrieve_k, rerank=not args.no_rerank)
        docs = [h.doc_id for h in res["hits"]]
        gold = set(case["gold_doc_ids"])
        hit = hit_at_k(docs, gold, args.k)
        rr = mrr_at_k(docs, gold, args.k)
        ndcg = ndcg_at_k(docs, gold, 10)
        r10 = hit_at_k(docs, gold, 10)
        hit_total += hit
        mrr_total += rr
        ndcg_total += ndcg
        recall10_total += r10
        rows.append({"query": case["query"], "got": docs, "gold": sorted(gold),
                     "hit@k": hit, "rr": round(rr, 3),
                     "ndcg@10": round(ndcg, 3), "recall@10": r10,
                     "top_score": (res["hits"][0].scores if res["hits"] else None),
                     "timings_ms": res["trace"].get("timings_ms")})
        mark = "PASS" if hit else "FAIL"
        print(f"{mark}  rr={rr:.2f} ndcg@10={ndcg:.2f}  {case['query'][:64]}")
        if not hit:
            print(f"      got {docs[:3]}… wanted {sorted(gold)}")

    n = len(cases)
    print(f"\nrecall@{args.k}: {hit_total}/{n} = {hit_total / n:.2%}   "
          f"MRR@{args.k}: {mrr_total / n:.3f}   "
          f"recall@10: {recall10_total}/{n} = {recall10_total / n:.2%}   "
          f"nDCG@10: {ndcg_total / n:.3f}")
    print(f"rerank={not args.no_rerank} depth={config.RERANK_DEPTH} "
          f"max_length={config.RERANK_MAX_LENGTH} retrieve_k={retrieve_k}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    out = RESULTS_DIR / f"retrieval_{util.iso_now().replace(':', '-')}{suffix}.json"
    out.write_text(json.dumps({"k": args.k, "retrieve_k": retrieve_k,
                               "rerank": not args.no_rerank,
                               "rerank_depth": config.RERANK_DEPTH,
                               "rerank_max_length": config.RERANK_MAX_LENGTH,
                               "queries_file": str(args.file),
                               "rows": rows, "recall": hit_total / n, "mrr": mrr_total / n,
                               "recall@10": recall10_total / n, "ndcg@10": ndcg_total / n},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
