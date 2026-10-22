#!/usr/bin/env python3
"""Build the persisted LangChain FAISS vector store from the data artifacts.

    python scripts/build_lc_store.py                # build data/index/lc_faiss/
    python scripts/build_lc_store.py --smoke "kyc"  # build + one similarity query

Reads data/chunks.jsonl + data/index/dense_fp16.npy and writes
data/index/lc_faiss/ (index.faiss + docstore). Same bge vectors as the
engine, so a LangChain retriever over the store returns the same dense
neighbours; the hybrid engine stays the primary path.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:  # Windows consoles default to cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="RegIntel LangChain FAISS store builder")
    ap.add_argument("--smoke", metavar="QUERY", help="run one similarity search after building")
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args(argv)

    from regintel.lc.vectorstore import build_vector_store, load_vector_store

    build_vector_store()
    if args.smoke:
        vs = load_vector_store()
        for doc, score in vs.similarity_search_with_score(args.smoke, k=args.k):
            m = doc.metadata
            print(f"  {score:+.4f}  {m.get('chunk_id')}  {m.get('title', '')[:60]}  "
                  f"para {m.get('para') or '-'}  p.{m.get('page_start')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
