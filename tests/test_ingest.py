"""Ingest-maintenance logic: refresh staleness detection and amendment linking."""
import json

from regintel import config, ingest


# ---------------------------------------------------------------------------
# refresh_backfill internals
# ---------------------------------------------------------------------------
def test_latest_by_page_picks_newest_version():
    docs = {
        "md1": {"doc_id": "md1", "source": "backfill", "page_id": 111,
                "fetched_at": "2026-01-01T00:00:00"},
        "md1-r20260814": {"doc_id": "md1-r20260814", "source": "backfill", "page_id": 111,
                          "fetched_at": "2026-08-14T00:00:00"},
        "nn7": {"doc_id": "nn7", "source": "rss", "page_id": 111,
                "fetched_at": "2026-09-01T00:00:00"},
    }
    latest = ingest._latest_by_page(docs)
    assert latest["111"]["doc_id"] == "md1-r20260814"  # rss docs never shadow backfill


def test_refresh_plan_flags_only_newer_listing_dates():
    cands = [
        {"pid": 1, "listing_title": "Some Direction (Updated as on August 14, 2026)"},
        {"pid": 2, "listing_title": "Direction Two (Updated as on June 1, 2026)"},
        {"pid": 3, "listing_title": "Direction Three"},  # no date -> unverifiable
        {"pid": 4, "listing_title": "Direction Four (Updated as on May 1, 2026)"},
    ]
    by_pid = {
        "1": {"doc_id": "md1", "last_updated": "2026-06-01", "fetched_at": "2026-06-02T00:00:00"},
        "2": {"doc_id": "md2", "last_updated": "2026-07-01", "fetched_at": "2026-07-02T00:00:00"},
        "3": {"doc_id": "md3", "last_updated": None, "fetched_at": "2026-07-02T00:00:00"},
        # pid 4 has no stored doc -> ignored
    }
    stale, unverifiable = ingest._refresh_plan(cands, by_pid)
    assert [c["pid"] for c, _, _ in stale] == [1]
    assert stale[0][1]["doc_id"] == "md1" and stale[0][2] == "2026-08-14"
    assert unverifiable == 1


# ---------------------------------------------------------------------------
# link_amendments
# ---------------------------------------------------------------------------
def test_norm_title_keeps_the_entity_parenthetical():
    a = ingest._norm_title("Reserve Bank of India (Commercial Banks - Prudential Norms "
                           "on Capital Adequacy) Twelfth Amendment Directions, 2026")
    b = ingest._norm_title("Reserve Bank of India (Commercial Banks - Prudential Norms "
                           "on Capital Adequacy) Directions, 2025")
    c = ingest._norm_title("Reserve Bank of India (Small Finance Banks - Prudential Norms "
                           "on Capital Adequacy) Directions, 2025")
    assert a == b
    assert a != c


def test_link_amendments_wires_amends_and_amended_by(tmp_path, monkeypatch):
    docs = [
        {"doc_id": "md1", "title": "Reserve Bank of India (Commercial Banks - Miscellaneous) "
                                   "Directions, 2025", "doc_type": "master_direction"},
        {"doc_id": "md2", "title": "Reserve Bank of India (Small Finance Banks - Miscellaneous) "
                                   "Directions, 2025", "doc_type": "master_direction"},
        {"doc_id": "nn9", "title": "Reserve Bank of India (Commercial Banks - Miscellaneous) "
                                   "Amendment Directions, 2026", "doc_type": "amendment_direction"},
        {"doc_id": "nn10", "title": "Unrelated Notice on Currency Chests",
         "doc_type": "notification"},
    ]
    chunks = [{"doc_id": "md1", "chunk_id": "c1"}, {"doc_id": "nn9", "chunk_id": "c2"}]
    doc_path, chunk_path = tmp_path / "documents.jsonl", tmp_path / "chunks.jsonl"
    doc_path.write_text("\n".join(json.dumps(d) for d in docs), encoding="utf-8")
    chunk_path.write_text("\n".join(json.dumps(c) for c in chunks), encoding="utf-8")
    monkeypatch.setattr(config, "DOCUMENTS", doc_path)
    monkeypatch.setattr(config, "CHUNKS", chunk_path)

    stats = ingest.link_amendments(progress=lambda *_: None)
    assert stats["linked"] == 1 and stats["details"][0]["target"] == "md1"

    out = {d["doc_id"]: d for d in
           (json.loads(l) for l in doc_path.read_text(encoding="utf-8").splitlines())}
    assert out["nn9"]["amends"] == ["md1"]
    assert out["md1"]["amended_by"] == ["nn9"]
    assert "amended_by" not in out["md2"]
    assert "amends" not in out["nn10"]

    out_chunks = {c["chunk_id"]: c for c in
                  (json.loads(l) for l in chunk_path.read_text(encoding="utf-8").splitlines())}
    assert out_chunks["c1"]["amended_by"] == ["nn9"]
    assert out_chunks["c2"]["amends"] == ["md1"]
