"""Artifact store: documents.jsonl, chunks.jsonl, manifest, sync log.

File-based by design — the whole corpus (text + chunks + indexes) travels in
the repository, the app only reads. Ingestion writes are append-only jsonl;
maintenance rewrites (``write_jsonl``) are atomic and preserve order, so the
positional dense index keeps matching chunks.jsonl. Rebuild commands
regenerate the indexes from chunks.jsonl at any time.
"""
from __future__ import annotations

import json
from pathlib import Path

from regintel import config, util


def append_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path: Path, records: list[dict]) -> None:
    """Atomic full rewrite (maintenance only: status recompute, refresh links).

    Normal ingestion stays append-only so the positional dense index keeps
    matching chunks.jsonl."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp.replace(path)


def load_documents() -> dict[str, dict]:
    return {d["doc_id"]: d for d in load_jsonl(config.DOCUMENTS)}


def load_chunks() -> list[dict]:
    return load_jsonl(config.CHUNKS)


def known_page_ids() -> set[str]:
    """Page ids across all stored documents (RSS + backfill dedupe)."""
    ids = set()
    for d in load_jsonl(config.DOCUMENTS):
        pid = d.get("page_id")
        if pid:
            ids.add(str(pid))
    return ids


def write_manifest(manifest: dict) -> str:
    manifest = dict(manifest)
    manifest["version"] = "v1-" + util.sha256_text(
        f"{manifest.get('n_chunks')}|{manifest.get('n_docs')}|{manifest.get('built_at')}")[:10]
    config.MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    config.MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest["version"]


def read_manifest() -> dict | None:
    if config.MANIFEST.exists():
        try:
            return json.loads(config.MANIFEST.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
    return None


def log_sync(entry: dict) -> None:
    entry = dict(entry, ts=util.iso_now())
    append_jsonl(config.SYNC_LOG, [entry])


def load_sync_log(limit: int = 50) -> list[dict]:
    return load_jsonl(config.SYNC_LOG)[-limit:]
