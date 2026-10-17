"""Ingest orchestrator — the single pipeline both the CLI and the app call.

    poll (RSS | backfill) -> diff vs store -> download -> layout-aware extract
        -> classify -> structure-aware chunk -> append artifacts
        -> rebuild BM25 + incrementally append dense vectors -> manifest

Every stage is idempotent: documents are keyed by RBI page id, so re-running
sync never duplicates a document. Failures are per-document and non-fatal.
"""
from __future__ import annotations

import difflib
import re
from pathlib import Path

import numpy as np

from regintel import (backfill, chunk as chunker, classify, config, extract, fetch, rss, store,
                      synclock, util)
from regintel.index.bm25 import BM25Index, tokenize
from regintel.index.dense import (DenseIndex, Encoder, append_vectors, dense_meta, fingerprint,
                                  prefix_matches)


def doc_id_for(pid: int, source: str) -> str:
    prefix = "nn" if source == "rss" else "md"
    return f"{prefix}{pid}"


def _denorm(doc: dict, chunks: list[dict]) -> list[dict]:
    """Copy doc-level metadata onto every chunk so retrieval filters and
    citation cards never need a join."""
    meta = {k: doc.get(k) for k in
            ("title", "reference", "issue_date", "last_updated", "as_of", "doc_type", "status",
             "effective_from", "supersedes", "superseded_by", "amends", "amended_by",
             "entities", "entity_scope", "themes", "content_class",
             "page_url", "pdf_url", "source")}
    for c in chunks:
        c.update(meta)
    return chunks


def process_document(meta: dict, progress=print, known_ids: set[str] | None = None,
                     extra: dict | None = None) -> dict | None:
    """meta: {doc_id, title, page_url, pdf_url, source, published, page_id}. Returns
    the stored document record or None when the doc is skipped. ``extra`` fields
    (e.g. supersedes=…) are merged into the record before it is written."""
    doc_id = meta["doc_id"]
    if known_ids is None:
        known_ids = set(store.load_documents())
    if doc_id in known_ids:
        progress(f"  = already in corpus: {doc_id}")
        return None
    progress(f"  downloading {meta['pdf_url']}")
    dl = fetch.download_pdf(meta["pdf_url"], doc_id)
    raw_path = config.RAW / f"{doc_id}.pdf"

    pages = extract.extract_pdf(raw_path)
    head = " ".join(p["text"] for p in pages[:2])[:4000]
    reference = meta.get("reference") or fetch.pdf_head_ref(raw_path)
    cls = classify.classify_document(meta["title"], head, reference)

    listing = meta.get("listing") or {}
    issue_date = (meta.get("published")
                  or listing.get("effective_from")
                  or util.parse_date(pages[0]["text"][:700]))
    effective_from = classify.effective_date("\n".join(p["text"] for p in pages))
    doc = {
        "doc_id": doc_id,
        "page_id": meta.get("page_id"),
        "source": meta["source"],
        "title": meta["title"],
        "page_url": meta["page_url"],
        "pdf_url": meta["pdf_url"],
        "reference": reference,
        "issue_date": issue_date,
        "last_updated": listing.get("last_updated"),
        "as_of": issue_date or listing.get("last_updated"),
        "status": classify.status_for(meta["title"], effective_from),
        "effective_from": effective_from,
        **cls,
        "sha256": dl["sha256"],
        "bytes": dl["bytes"],
        "n_pages": len(pages),
        "fetched_at": util.iso_now(),
    }
    if extra:
        doc.update(extra)
    chunks = chunker.build_chunks(doc_id, pages)
    if not chunks:
        progress(f"  ! no chunks produced for {doc_id} — skipped")
        return None
    doc["n_chunks"] = len(chunks)
    doc["pages"] = pages
    _denorm(doc, chunks)
    store.append_jsonl(config.DOCUMENTS, [doc])
    store.append_jsonl(config.CHUNKS, chunks)
    progress(f"  + {doc_id}: {len(pages)} pages -> {len(chunks)} chunks "
             f"[{', '.join(cls['entities']) or 'broad'}]")
    return doc


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
def sync_rss(limit: int | None = None, progress=print) -> dict:
    items = rss.fetch_items()
    seen = rss.load_seen()
    todo = rss.new_items(items, seen)
    progress(f"RSS: {len(items)} items in feed, {len(todo)} new")
    stats = {"new_docs": 0, "failures": 0, "details": []}
    known_ids = set(store.load_documents())
    for item in todo[:limit] if limit else todo:
        pid, title = item["pid"], util.clean_title(item["title"])
        doc_id = doc_id_for(pid, "rss")
        if doc_id in known_ids:  # already ingested from a previous run
            seen["pages"][str(pid)] = {"title": title, "status": "ingested", "doc_id": doc_id}
            continue
        try:
            pdf_url, ref = fetch.find_pdf(item["link"])
            if not pdf_url:
                progress(f"  - no PDF on page {pid} ({title[:70]}) — marked seen")
                seen["pages"][str(pid)] = {"title": title, "status": "no_pdf"}
                continue
            doc = process_document({
                "doc_id": doc_id_for(pid, "rss"), "title": title,
                "page_url": item["link"], "pdf_url": pdf_url, "reference": ref,
                "source": "rss", "published": item["published"], "page_id": pid,
            }, progress, known_ids)
            if doc:
                stats["new_docs"] += 1
                stats["details"].append(doc["doc_id"])
            seen["pages"][str(pid)] = {"title": title, "status": "ingested",
                                       "doc_id": doc_id_for(pid, "rss")}
        except Exception as exc:  # noqa: BLE001
            progress(f"  ! {pid} failed: {type(exc).__name__}: {str(exc)[:160]}")
            stats["failures"] += 1
            seen["pages"][str(pid)] = {"title": title, "status": "failed",
                                       "error": f"{type(exc).__name__}"}
        rss.save_seen(seen)
    seen.setdefault("pages", {})
    rss.save_seen(seen)
    return stats


def sync_backfill(limit: int | None = None, progress=print) -> dict:
    cands = backfill.crawl_all()
    known = store.known_page_ids()
    todo = [c for c in cands if str(c["pid"]) not in known]
    progress(f"Backfill: {len(cands)} listing entries, {len(todo)} new")
    stats = {"new_docs": 0, "failures": 0, "details": []}
    known_ids = set(store.load_documents())
    for cand in todo[:limit] if limit else todo:
        pid = cand["pid"]
        title = util.clean_title(cand["listing_title"])
        try:
            view = backfill.resolve_view(pid)
            if not view["pdf_url"]:
                progress(f"  - no PDF for view page {pid} ({title[:70]})")
                stats["failures"] += 1
                continue
            doc = process_document({
                "doc_id": doc_id_for(pid, "backfill"), "title": title,
                "page_url": view["page_url"], "pdf_url": view["pdf_url"],
                "source": "backfill", "published": None, "page_id": pid,
                "listing": backfill.listing_meta(cand["listing_title"]),
            }, progress, known_ids)
            if doc:
                stats["new_docs"] += 1
                stats["details"].append(doc["doc_id"])
        except Exception as exc:  # noqa: BLE001
            progress(f"  ! {pid} failed: {type(exc).__name__}: {str(exc)[:160]}")
            stats["failures"] += 1
    return stats


def _mark_superseded(flipped: dict[str, str], progress=print) -> int:
    """Flip previous versions of refreshed documents to status 'superseded'.

    Retrieval hides withdrawn and superseded documents unless the caller asks
    for them; the old record and its chunks stay in the store for the audit
    trail. ``flipped``: old doc_id -> replacement doc_id.
    """
    if not flipped:
        return 0
    docs = store.load_jsonl(config.DOCUMENTS)
    for d in docs:
        new_id = flipped.get(d["doc_id"])
        if new_id:
            d["status"] = "superseded"
            d["superseded_by"] = new_id
    store.write_jsonl(config.DOCUMENTS, docs)
    chunks = store.load_jsonl(config.CHUNKS)
    n = 0
    for c in chunks:
        new_id = flipped.get(c["doc_id"])
        if new_id:
            c["status"] = "superseded"
            c["superseded_by"] = new_id
            n += 1
    store.write_jsonl(config.CHUNKS, chunks)
    progress(f"  {len(flipped)} previous version(s) marked superseded ({n} chunks)")
    return n


def _latest_by_page(docs: dict[str, dict]) -> dict[str, dict]:
    """Newest stored backfill version per RBI page id (by fetched_at)."""
    out: dict[str, dict] = {}
    for d in docs.values():
        if d.get("source") == "backfill" and d.get("page_id"):
            cur = out.get(str(d["page_id"]))
            if cur is None or (d.get("fetched_at") or "") > (cur.get("fetched_at") or ""):
                out[str(d["page_id"])] = d
    return out


def _refresh_plan(cands: list[dict], by_pid: dict[str, dict]
                  ) -> tuple[list[tuple[dict, dict, str]], int]:
    """Candidates whose listing ('Updated as on …') is newer than the stored doc.

    Returns ([(candidate, stored_doc, new_date)], unverifiable). Pages with a
    missing listing date on either side cannot be compared and are counted as
    unverifiable.
    """
    stale: list[tuple[dict, dict, str]] = []
    unverifiable = 0
    for cand in cands:
        cur = by_pid.get(str(cand["pid"]))
        if cur is None:
            continue
        new_upd = backfill.listing_meta(cand["listing_title"]).get("last_updated")
        old_upd = cur.get("last_updated")
        if not new_upd or not old_upd:
            unverifiable += 1
            continue
        if new_upd > old_upd:
            stale.append((cand, cur, new_upd))
    return stale, unverifiable


def refresh_backfill(progress=print, dry_run: bool = False) -> dict:
    """Re-check the Master Directions listings for in-place updates.

    RBI amends a consolidated Direction by replacing the PDF behind the same
    view id and bumping "(Updated as on …)" in the listing title. A listed page
    whose listing date is newer than the stored one is re-ingested as a new
    version (doc_id `<base>-r<date>`); the previous version is marked
    superseded and hidden from retrieval by default. Pages without a listing
    date on either side cannot be compared and are counted as unverifiable.
    """
    cands = backfill.crawl_all()
    docs = store.load_documents()
    by_pid = _latest_by_page(docs)
    plan, unverifiable = _refresh_plan(cands, by_pid)
    stats = {"checked": len({str(c["pid"]) for c in cands if str(c["pid"]) in by_pid}),
             "unverifiable": unverifiable, "stale": len(plan), "refreshed": 0,
             "failed": 0, "details": []}
    flipped: dict[str, str] = {}
    for cand, cur, new_upd in plan:
        new_id = f"{cur['doc_id']}-r{new_upd.replace('-', '')}"
        title = util.clean_title(cand["listing_title"])
        progress(f"  * {cand['pid']} updated on {new_upd} (stored: {cur.get('last_updated')}) "
                 f"-> {new_id}")
        if dry_run:
            stats["details"].append({"page_id": cand["pid"], "old": cur["doc_id"],
                                     "new": new_id, "listing_date": new_upd})
            continue
        try:
            if new_id in docs:  # already refreshed in an earlier run
                flipped[cur["doc_id"]] = new_id
                stats["refreshed"] += 1
                continue
            view = backfill.resolve_view(cand["pid"])
            if not view["pdf_url"]:
                raise RuntimeError("no PDF on the view page")
            doc = process_document({
                "doc_id": new_id, "title": title, "page_url": view["page_url"],
                "pdf_url": view["pdf_url"], "source": "backfill",
                "published": None, "page_id": cand["pid"],
                "listing": backfill.listing_meta(cand["listing_title"]),
            }, progress, set(docs), extra={"supersedes": cur["doc_id"]})
            if doc:
                docs[doc["doc_id"]] = doc
                flipped[cur["doc_id"]] = doc["doc_id"]
                stats["refreshed"] += 1
                stats["details"].append({"page_id": cand["pid"], "old": cur["doc_id"],
                                         "new": doc["doc_id"], "listing_date": new_upd})
        except Exception as exc:  # noqa: BLE001
            progress(f"  ! refresh failed for page {cand['pid']}: "
                     f"{type(exc).__name__}: {str(exc)[:160]}")
            stats["failed"] += 1
    if not dry_run:
        _mark_superseded(flipped, progress)
    return stats


# ---------------------------------------------------------------------------
# Amendment linking (title-based pointers, not legal consolidation)
# ---------------------------------------------------------------------------
_ORDINAL_RE = re.compile(
    r"\b(?:\d{1,2}(?:st|nd|rd|th)|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|"
    r"tenth|eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|"
    r"eighteenth|nineteenth|twentieth|twenty-first|twenty-second)\b", re.I)
_AMENDMENT_TITLE_RE = re.compile(r"\bamend(?:ment|ments|ed|s|ing)?\b", re.I)


def _norm_title(title: str) -> str:
    """Title key for amendment matching: drop amendment/ordinal/year noise but
    keep the entity parenthetical (it is what distinguishes the ~10 versions)."""
    t = util.clean_title(title).lower()
    t = _AMENDMENT_TITLE_RE.sub(" ", t)
    t = _ORDINAL_RE.sub(" ", t)
    t = re.sub(r"\b20\d\d\b", " ", t)
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def link_amendments(progress=print) -> dict:
    """Link amendment notices to the Master Directions they amend (title match).

    The link is a pointer for the citation card ("amends: …" / "amended by:
    …") — the amended paragraphs are NOT merged into the parent text. Exact
    consolidation of amending clause language is out of scope for this build.
    """
    docs = store.load_jsonl(config.DOCUMENTS)
    by_id = {d["doc_id"]: d for d in docs}
    targets = [(d["doc_id"], _norm_title(d["title"])) for d in docs
               if d.get("doc_type") == "master_direction"]
    stats = {"amendment_docs": 0, "linked": 0, "updated": 0, "details": []}
    reverse: dict[str, list[str]] = {}
    for d in docs:
        if not _AMENDMENT_TITLE_RE.search(d.get("title") or ""):
            continue
        stats["amendment_docs"] += 1
        key = _norm_title(d["title"])
        best_id, best_ratio = None, 0.0
        for tid, tkey in targets:
            if tid == d["doc_id"]:
                continue
            ratio = difflib.SequenceMatcher(None, key, tkey).ratio()
            if ratio > best_ratio:
                best_id, best_ratio = tid, ratio
        if best_id and best_ratio >= 0.72:
            if d.get("amends") != [best_id]:
                d["amends"] = [best_id]
                stats["updated"] += 1
            stats["linked"] += 1
            reverse.setdefault(best_id, []).append(d["doc_id"])
            stats["details"].append({"amendment": d["doc_id"], "target": best_id,
                                     "ratio": round(best_ratio, 3)})
    for tid, amenders in reverse.items():
        cur = by_id[tid].get("amended_by") or []
        want = sorted(set(cur) | set(amenders))
        if want != cur:
            by_id[tid]["amended_by"] = want
            stats["updated"] += 1
    if stats["updated"]:
        store.write_jsonl(config.DOCUMENTS, docs)
        link_by_id = {d["doc_id"]: {"amends": d.get("amends"),
                                    "amended_by": d.get("amended_by")} for d in docs}
        chunks = store.load_jsonl(config.CHUNKS)
        for c in chunks:
            info = link_by_id.get(c["doc_id"])
            if info:
                c["amends"] = info["amends"]
                c["amended_by"] = info["amended_by"]
        store.write_jsonl(config.CHUNKS, chunks)
        progress(f"  amendment links updated: {stats['linked']} linked, "
                 f"{stats['updated']} records rewritten")
    return stats


# ---------------------------------------------------------------------------
# Index build
# ---------------------------------------------------------------------------
def rebuild_indexes(progress=print, full_dense: bool = False) -> dict:
    chunks = store.load_chunks()
    if not chunks:
        raise RuntimeError("no chunks to index — run a sync first")
    texts = [c["text"] for c in chunks]

    progress(f"BM25: building over {len(texts)} chunks…")
    bm = BM25Index.build([tokenize(t) for t in texts])
    bm.save(config.INDEX_DIR)

    encoder = Encoder()
    meta = dense_meta(config.INDEX_DIR)
    n_prev = int(meta["n"]) if meta else 0
    npy = config.INDEX_DIR / "dense_fp16.npy"
    # Reuse / append only when the stored vectors provably belong to the first
    # n_prev chunks of the current corpus (fingerprint over chunk_id + text_hash)
    # AND were produced by the same encoder. A changed REGINTEL_EMBED (or config
    # default) would otherwise append vectors from a different embedding space —
    # same dimension, incomparable geometry — and relabel the whole index.
    stored_encoder = (meta or {}).get("encoder")
    same_encoder = stored_encoder == encoder.model_name
    aligned = (bool(meta) and same_encoder and prefix_matches(meta.get("fingerprint"), chunks)
               and meta["fingerprint"].get("n") == n_prev <= len(texts)
               and npy.exists() and np.load(npy, mmap_mode="r").shape[0] == n_prev)
    if full_dense or not aligned:
        if full_dense or not meta:
            why = ""
        elif not same_encoder:
            why = (f" (encoder changed: index was built with {stored_encoder!r}, "
                   f"now {encoder.model_name!r})")
        else:
            why = " (stored vectors are not verifiably aligned with chunks.jsonl)"
        progress(f"Dense: full encode of {len(texts)} chunks with {encoder.model_name}{why}…")
        vecs = encoder.encode_passages(texts)
    elif n_prev == len(texts):
        progress("Dense: index already matches the corpus — reused")
        vecs = np.load(npy)
    else:
        progress(f"Dense: incremental encode of {len(texts) - n_prev} new chunks "
                 f"(reusing {n_prev})…")
        vecs = append_vectors(npy, encoder, texts[n_prev:], expect_n=n_prev)
    DenseIndex(vecs).save(config.INDEX_DIR, {"encoder": encoder.model_name,
                                             "fingerprint": fingerprint(chunks)})

    docs = store.load_documents()
    manifest = {
        "built_at": util.iso_now(),
        "n_docs": len(docs),
        "n_chunks": len(chunks),
        "sources": {s: sum(1 for d in docs.values() if d["source"] == s)
                    for s in ("rss", "backfill")},
        "embed_model": encoder.model_name,
        "chunk_words": [config.CHUNK_MIN_WORDS, config.CHUNK_MAX_WORDS],
        "docs_types": {},
    }
    for d in docs.values():
        manifest["docs_types"][d["doc_type"]] = manifest["docs_types"].get(d["doc_type"], 0) + 1
    version = store.write_manifest(manifest)
    progress(f"manifest {version}: {manifest['n_docs']} docs / {manifest['n_chunks']} chunks")
    return {**manifest, "version": version}


def check_updates() -> dict:
    """Cheap feed check for the UI: how many unseen notifications are live now."""
    try:
        items = rss.fetch_items()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:120]}", "new": 0}
    seen = rss.load_seen()
    new = rss.new_items(items, seen)
    return {"ok": True, "new": len(new),
            "items": [{"title": i["title"], "published": i["published"], "pid": i["pid"]}
                      for i in new]}


def sync(source: str = "rss", limit: int | None = None, rebuild: bool = True,
         refresh: bool = False, progress=print) -> dict:
    """One full pass, under an inter-process lock (see synclock): if another
    sync is running this one is skipped, not queued — the result carries
    ``skipped: True`` and the reason. source: rss | backfill | both.
    """
    try:
        with synclock.sync_lock() as lock:
            return _sync(source, limit, rebuild, refresh, lock.wrap(progress))
    except synclock.SyncBusy as exc:
        progress(f"Sync skipped: {exc}")
        return {"skipped": True, "reason": str(exc), "rss": {"new_docs": 0},
                "backfill": {"new_docs": 0}, "links": {}, "failures": 0}


def _sync(source: str, limit: int | None, rebuild: bool, refresh: bool, progress) -> dict:
    """The sync pass itself.

    ``refresh=True`` additionally re-checks the Master Directions listings for
    in-place updates (RBI bumps "Updated as on"; changed pages are re-ingested
    as new versions). Amendment notices are linked to their parent Directions
    after ingestion (title-based pointers — see link_amendments)."""
    result = {"rss": {"new_docs": 0}, "backfill": {"new_docs": 0}, "failures": 0}
    if source in ("rss", "both"):
        result["rss"] = sync_rss(limit=limit, progress=progress)
    if source in ("backfill", "both"):
        result["backfill"] = sync_backfill(limit=limit, progress=progress)
    if refresh:
        progress("Refresh: re-checking listings for in-place updates…")
        result["refresh"] = refresh_backfill(progress=progress)
    result["links"] = link_amendments(progress=progress)
    result["failures"] = (result["rss"].get("failures", 0)
                          + result["backfill"].get("failures", 0)
                          + result.get("refresh", {}).get("failed", 0))
    new_total = (result["rss"].get("new_docs", 0) + result["backfill"].get("new_docs", 0)
                 + result.get("refresh", {}).get("refreshed", 0))
    if rebuild and new_total:
        progress("Rebuilding indexes…")
        result["manifest"] = rebuild_indexes(progress=progress)
    store.log_sync({"source": source, "new_docs": new_total, "failures": result["failures"],
                    "refreshed": result.get("refresh", {}).get("refreshed", 0),
                    "amendment_links": result["links"].get("linked", 0),
                    "doc_ids": result["rss"].get("details", []) + result["backfill"].get("details", [])})
    return result
