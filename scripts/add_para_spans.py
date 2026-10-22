#!/usr/bin/env python3
"""Add ``para_end`` metadata to chunks.jsonl without touching the dense index.

The chunker records the paragraph label a passage opens at (``para``).  This
script re-derives the chunking from the stored pages and records the *last*
label merged into each passage (``para_end``), so citations can say
"Paras 4.2-4.4" instead of only "Para 4.2".

It is metadata-only: chunk ids, text and text hashes must match the existing
chunks exactly, so ``dense_meta.json``'s fingerprint stays valid and no
re-encoding is needed.  Any mismatch aborts the run without writing.

    python scripts/add_para_spans.py            # dry run
    python scripts/add_para_spans.py --write    # update chunks.jsonl
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:  # Windows consoles default to cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

from regintel import chunk as chunker, config, store  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="write chunks.jsonl (default: dry run)")
    args = ap.parse_args()

    docs = store.load_documents()
    chunks = store.load_chunks()
    if not docs or not chunks:
        print("no documents/chunks stored — run a sync first")
        return 1

    by_doc: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        by_doc[c["doc_id"]].append(c)

    regenerated: dict[str, list[dict]] = {}
    problems: list[str] = []
    for doc_id, old in by_doc.items():
        pages = (docs.get(doc_id) or {}).get("pages") or []
        if not pages:
            problems.append(f"{doc_id}: document has no stored pages")
            continue
        new = chunker.build_chunks(doc_id, pages)
        if len(new) != len(old):
            problems.append(f"{doc_id}: chunk count {len(old)} -> {len(new)}")
            continue
        for o, n in zip(old, new):
            if o["chunk_id"] != n["chunk_id"] or o["text_hash"] != n["text_hash"]:
                problems.append(f"{doc_id}: row {o['chunk_id']} does not regenerate "
                                f"({n['chunk_id']}, {o['text_hash']} -> {n['text_hash']})")
                break
            if (o.get("para") or "") != (n.get("para") or ""):
                problems.append(f"{doc_id}: {o['chunk_id']} para label changed "
                                f"{o.get('para')!r} -> {n.get('para')!r}")
                break
        else:
            regenerated[doc_id] = new

    if problems:
        print(f"ABORT — {len(problems)} document(s) do not regenerate identically:")
        for p in problems[:20]:
            print("  ", p)
        print("chunks.jsonl was not modified; the dense index is still aligned.")
        return 1

    by_id = {doc_id: {n["chunk_id"]: n for n in new}
             for doc_id, new in regenerated.items()}
    changed = 0
    spans = 0
    for c in chunks:
        n = by_id[c["doc_id"]][c["chunk_id"]]
        end = n.get("para_end") or n.get("para") or ""
        if c.get("para_end") != end:
            changed += 1
        c["para_end"] = end
        if end and end != (c.get("para") or ""):
            spans += 1

    print(f"{len(regenerated)} docs / {len(chunks)} chunks regenerate identically")
    print(f"para_end set on {len(chunks) - sum(1 for c in chunks if not c.get('para_end'))} chunks; "
          f"{spans} span more than one paragraph label; {changed} changed")
    if not args.write:
        print("dry run — pass --write to update chunks.jsonl (no re-index needed)")
        return 0

    store.write_jsonl(config.CHUNKS, chunks)
    print(f"wrote {config.CHUNKS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
