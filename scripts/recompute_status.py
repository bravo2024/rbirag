#!/usr/bin/env python3
"""Recompute document status + effective_from across the stored corpus.

Maintenance pass for classification changes (forthcoming detection from
commencement dates): re-derives status and effective_from from the stored
pages of every document, updates documents.jsonl and the denormalised status
on chunks.jsonl (atomic rewrites, order preserved), then optionally rebuilds
the indexes.

    python scripts/recompute_status.py --dry-run
    python scripts/recompute_status.py --rebuild
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:  # Windows consoles default to cp1252; output may contain unicode
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

from regintel import classify, config, ingest, store  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Recompute corpus statuses")
    ap.add_argument("--dry-run", action="store_true", help="report changes only")
    ap.add_argument("--rebuild", action="store_true", help="rebuild indexes afterwards")
    args = ap.parse_args(argv)

    docs = store.load_jsonl(config.DOCUMENTS)
    changes = []
    meta_by_id: dict[str, dict] = {}
    for d in docs:
        text = "\n".join(p.get("text") or "" for p in d.get("pages") or [])
        eff = classify.effective_date(text)
        status = classify.status_for(d.get("title") or "", eff)
        if status != d.get("status") or eff != d.get("effective_from"):
            changes.append((d["doc_id"], d.get("status"), status, eff, (d.get("title") or "")[:74]))
        d["status"] = status
        d["effective_from"] = eff
        meta_by_id[d["doc_id"]] = {"status": status, "effective_from": eff}

    print(f"documents: {len(docs)} | changed: {len(changes)}")
    for doc_id, old, new, eff, title in changes:
        print(f"  {doc_id}: {old} -> {new}"
              + (f" (effective {eff})" if eff else "") + f" | {title}")

    if args.dry_run:
        print("dry run — nothing written")
        return 0

    store.write_jsonl(config.DOCUMENTS, docs)
    chunks = store.load_jsonl(config.CHUNKS)
    n_chunk_updates = 0
    for c in chunks:
        meta = meta_by_id.get(c["doc_id"])
        if meta and (c.get("status") != meta["status"]
                     or c.get("effective_from") != meta["effective_from"]):
            c["status"] = meta["status"]
            c["effective_from"] = meta["effective_from"]
            n_chunk_updates += 1
    store.write_jsonl(config.CHUNKS, chunks)
    print(f"chunks updated: {n_chunk_updates}")
    if args.rebuild:
        manifest = ingest.rebuild_indexes()
        print(f"index: {manifest['n_docs']} docs / {manifest['n_chunks']} chunks "
              f"-> {manifest['version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
