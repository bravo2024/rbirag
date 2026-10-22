#!/usr/bin/env python3
"""Rebuild the retrieval indexes from data/chunks.jsonl (BM25 + dense).

    python scripts/build_index.py             # incremental (reuse dense prefix)
    python scripts/build_index.py --full      # full dense re-encode
    python scripts/build_index.py --bm25-only # BM25 refresh only

This is the deploy path: a fresh clone with committed data/ artifacts can be
rebuilt without touching the network.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:  # Windows consoles default to cp1252; progress messages contain unicode
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

from regintel import config, ingest, store  # noqa: E402
from regintel.index.bm25 import BM25Index, tokenize  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="RegIntel index builder")
    ap.add_argument("--full", action="store_true", help="full dense re-encode")
    ap.add_argument("--bm25-only", action="store_true")
    args = ap.parse_args(argv)

    if args.bm25_only:
        chunks = store.load_chunks()
        bm = BM25Index.build([tokenize(c["text"]) for c in chunks])
        bm.save(config.INDEX_DIR)
        print(f"bm25 rebuilt over {len(chunks)} chunks")
        return 0

    manifest = ingest.rebuild_indexes(full_dense=args.full)
    print(f"{manifest['n_docs']} docs / {manifest['n_chunks']} chunks -> {manifest['version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
