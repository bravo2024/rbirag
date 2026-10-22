#!/usr/bin/env python3
"""Re-derive data/chunks.jsonl from the parsed documents (no network).

Useful after a chunker change: documents.jsonl stores every page's extracted
text, so chunks can be regenerated deterministically, then indexed:

    python scripts/rechunk.py && python scripts/build_index.py --full
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:  # Windows consoles default to cp1252; progress messages contain unicode
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

from regintel import chunk as chunker, config, ingest, store  # noqa: E402


def main() -> int:
    docs = store.load_documents()
    if not docs:
        print("no documents stored — run a sync first")
        return 1
    everything: list[dict] = []
    updated_docs: list[dict] = []
    for doc in docs.values():
        doc["as_of"] = doc.get("as_of") or doc.get("issue_date") or doc.get("last_updated")
        chunks = chunker.build_chunks(doc["doc_id"], doc.get("pages") or [])
        ingest._denorm(doc, chunks)
        doc["n_chunks"] = len(chunks)
        updated_docs.append(doc)
        everything.extend(chunks)
        print(f"{doc['doc_id']:>8}: {len(chunks):>4} chunks")

    config.DOCUMENTS.write_text(
        "".join(json.dumps(d, ensure_ascii=False, separators=(",", ":")) + "\n"
                for d in updated_docs), encoding="utf-8")
    config.CHUNKS.write_text(
        "".join(json.dumps(c, ensure_ascii=False, separators=(",", ":")) + "\n"
                for c in everything), encoding="utf-8")
    print(f"rechunked: {len(updated_docs)} docs -> {len(everything)} chunks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
