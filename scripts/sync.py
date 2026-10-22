#!/usr/bin/env python3
"""RegIntel ingestion CLI.

    python scripts/sync.py --source rss               # poll the live RSS feed
    python scripts/sync.py --source backfill          # crawl the MD listings
    python scripts/sync.py --source both --limit 20   # cap docs per source
    python scripts/sync.py --rebuild                  # full dense re-encode

Every run: fetch new documents -> parse -> classify -> chunk -> append ->
rebuild BM25 + incrementally extend the dense index -> bump the manifest.
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

from regintel import ingest  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="RegIntel corpus sync")
    ap.add_argument("--source", choices=["rss", "backfill", "both"], default="rss")
    ap.add_argument("--limit", type=int, default=None, help="max new docs per source")
    ap.add_argument("--refresh", action="store_true",
                    help="re-check MD listings for in-place updates "
                         "(re-ingests changed pages as new versions)")
    ap.add_argument("--no-index", action="store_true", help="skip index rebuild")
    ap.add_argument("--rebuild", action="store_true", help="full dense re-encode, no fetch")
    args = ap.parse_args(argv)

    if args.rebuild:
        manifest = ingest.rebuild_indexes(full_dense=True)
        print(f"rebuilt: {manifest['n_docs']} docs / {manifest['n_chunks']} chunks "
              f"-> {manifest['version']}")
        return 0

    result = ingest.sync(source=args.source, limit=args.limit, refresh=args.refresh,
                         rebuild=not args.no_index, progress=print)
    if result.get("skipped"):
        print(f"\nskipped: {result['reason']}")
        return 0
    new = result["rss"].get("new_docs", 0) + result["backfill"].get("new_docs", 0)
    refreshed = result.get("refresh", {}).get("refreshed", 0)
    links = result.get("links", {}).get("linked", 0)
    print(f"\ndone: {new} new documents, {refreshed} refreshed, {links} amendment links, "
          f"{result['failures']} failures")
    if "manifest" in result:
        m = result["manifest"]
        print(f"index: {m['n_docs']} docs / {m['n_chunks']} chunks -> {m['version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
