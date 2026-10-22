#!/usr/bin/env python3
"""A/B two embedding models on the same corpus, BM25 arm and query sets.

    python scripts/eval_encoder_ab.py --a data/index --b data/index_hinglish \
        --files evals/queries.jsonl evals/queries_holdout.jsonl

For each index it reports document-level recall@5, recall@10, MRR@5 and
nDCG@10 for:
  * dense-only   — the encoder's own ranking (no BM25, no rerank)
  * hybrid       — BM25 + dense, RRF fusion (no rerank)
  * hybrid+rerank — same, with the shared cross-encoder (``--rerank``)

The corpus and BM25 are shared, so any difference is attributable to the
encoder. Each index's query encoder is read from its ``dense_meta.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
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
from regintel.index.dense import DenseIndex, Encoder, dense_meta  # noqa: E402

RESULTS_DIR = ROOT / "evals" / "results"
K = 10


def load_sets(paths: list[str]) -> dict[str, list[dict]]:
    out = {}
    for p in paths:
        path = Path(p)
        out[path.stem] = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
                          if l.strip()]
    return out


def _filters(case: dict) -> QueryFilters | None:
    inst = case.get("institution")
    if not inst:
        return None
    return QueryFilters(entities=config.INSTITUTIONS[inst]["entities"])


def _dedupe(docs: list[str]) -> list[str]:
    seen, out = set(), []
    for d in docs:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def _row(query: str, docs: list[str], gold: set[str], **extra) -> dict:
    docs = _dedupe(docs)
    return {"query": query, "docs": docs, "gold": sorted(gold),
            "hit@5": hit_at_k(docs, gold, 5), "recall@10": hit_at_k(docs, gold, 10),
            "rr@5": round(mrr_at_k(docs, gold, 5), 4),
            "ndcg@10": round(ndcg_at_k(docs, gold, 10), 4), **extra}


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    return {"n": n,
            "recall@5": round(sum(r["hit@5"] for r in rows) / n, 4),
            "recall@10": round(sum(r["recall@10"] for r in rows) / n, 4),
            "mrr@5": round(sum(r["rr@5"] for r in rows) / n, 4),
            "ndcg@10": round(sum(r["ndcg@10"] for r in rows) / n, 4)}


def evaluate(index_dir: Path, sets: dict[str, list[dict]], rerank: bool) -> dict:
    meta = dense_meta(index_dir) or {}
    encoder_name = meta.get("encoder") or config.EMBED_MODEL
    chunks = store.load_chunks()
    dense = DenseIndex.load(index_dir)
    encoder = Encoder(model=encoder_name)
    engine = Engine(index_dir=index_dir)
    if not engine.ready:
        raise SystemExit(f"index not ready: {index_dir}")
    print(f"\n{index_dir} | encoder={encoder_name} | n={dense.n}", flush=True)

    result = {"dir": str(index_dir), "encoder": encoder_name,
              "n_vectors": int(dense.n), "sets": {}}
    for set_name, cases in sets.items():
        rows = {"dense": [], "hybrid": []}
        if rerank:
            rows["hybrid_rerank"] = []
        for case in cases:
            q, gold, filt = case["query"], set(case["gold_doc_ids"]), _filters(case)

            # dense-only
            mask = filt.mask(chunks) if filt else None
            qvec = encoder.encode_queries([q])[0]
            pairs = dense.search(qvec, k=K, mask=mask)
            rows["dense"].append(_row(q, [chunks[i]["doc_id"] for i, _ in pairs], gold))

            # hybrid RRF, no rerank
            t = time.perf_counter()
            r = engine.retrieve(q, filters=filt, k=K, rerank=False)
            rows["hybrid"].append(_row(q, [h.doc_id for h in r["hits"]], gold,
                                       ms=round((time.perf_counter() - t) * 1000)))
            if rerank:
                t = time.perf_counter()
                r = engine.retrieve(q, filters=filt, k=K, rerank=True)
                rows["hybrid_rerank"].append(_row(q, [h.doc_id for h in r["hits"]], gold,
                                                  ms=round((time.perf_counter() - t) * 1000)))
        result["sets"][set_name] = {
            "summary": {m: summarize(rs) for m, rs in rows.items()},
            "rows": rows}
        for method, rs in rows.items():
            s = result["sets"][set_name]["summary"][method]
            print(f"  {set_name:8s} {method:13s} n={s['n']:2d} "
                  f"R@5={s['recall@5']:.3f} R@10={s['recall@10']:.3f} "
                  f"MRR@5={s['mrr@5']:.3f} nDCG@10={s['ndcg@10']:.3f}", flush=True)
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="A/B embedding models on one corpus")
    ap.add_argument("--a", default=str(ROOT / "data" / "index"), help="index dir A (baseline)")
    ap.add_argument("--b", required=True, help="index dir B (challenger)")
    ap.add_argument("--files", nargs="+", default=[str(ROOT / "evals" / "queries.jsonl")])
    ap.add_argument("--rerank", action="store_true", help="also run the hybrid+cross-encoder arm")
    ap.add_argument("--tag", default="", help="suffix for the result file")
    args = ap.parse_args(argv)

    sets = load_sets(args.files)
    manifest = store.read_manifest() or {}
    payload = {"corpus": {"n_chunks": len(store.load_chunks()),
                          "n_docs": len(store.load_documents()),
                          "version": manifest.get("version")},
               "k": K, "files": args.files, "models": {}}
    for label, path in (("a", Path(args.a)), ("b", Path(args.b))):
        payload["models"][label] = evaluate(path, sets, rerank=args.rerank)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    out = RESULTS_DIR / f"encoder_ab_{util.iso_now().replace(':', '-')}{suffix}.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # markdown summary
    lines = [f"# Encoder A/B — {util.iso_now()}", "",
             f"Corpus: {payload['corpus']['n_docs']} docs / {payload['corpus']['n_chunks']} chunks "
             f"(`{payload['corpus']['version']}`)", "",
             "| model | encoder | set | method | n | recall@5 | recall@10 | MRR@5 | nDCG@10 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for label, m in payload["models"].items():
        for set_name, s in m["sets"].items():
            for method, sm in s["summary"].items():
                lines.append(f"| {label} | `{m['encoder']}` | {set_name} | {method} | {sm['n']} | "
                             f"{sm['recall@5']:.3f} | {sm['recall@10']:.3f} | {sm['mrr@5']:.3f} | "
                             f"{sm['ndcg@10']:.3f} |")
    md = out.with_suffix(".md")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {out.relative_to(ROOT)}")
    print(f"wrote {md.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
