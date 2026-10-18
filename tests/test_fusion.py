"""RRF fusion, dedupe, filters and the abstain decision."""
from types import SimpleNamespace

from regintel.index.fusion import QueryFilters, abstain_decision, dedupe_fused, rrf_fuse

CHUNKS = [
    {"chunk_id": "a", "text_hash": "h1", "content_class": "regulation", "status": "current",
     "issue_date": "2026-01-01", "doc_type": "master_direction", "themes": ["KYC / AML"],
     "entities": ["nbfc"], "entity_scope": "specific"},
    {"chunk_id": "b", "text_hash": "h2", "content_class": "regulation", "status": "current",
     "issue_date": "2024-01-01", "doc_type": "circular", "themes": [],
     "entities": ["commercial_banks"], "entity_scope": "specific"},
    {"chunk_id": "c", "text_hash": "h3", "content_class": "regulation", "status": "withdrawn",
     "issue_date": "2025-01-01", "doc_type": "circular", "themes": [],
     "entities": [], "entity_scope": "broad"},
    {"chunk_id": "d", "text_hash": "h4", "content_class": "market_ops", "status": "current",
     "issue_date": None, "doc_type": "notification", "themes": [],
     "entities": [], "entity_scope": "broad"},
]


def test_rrf_order():
    fused = rrf_fuse([[0, 1, 2], [2, 0, 1]])
    order = [i for i, _ in fused]
    assert order == [0, 2, 1]


def test_dedupe_keeps_best():
    fused = [(0, 0.03), (1, 0.02), (2, 0.01)]
    chunks = [{"chunk_id": "a", "text_hash": "same"}, {"chunk_id": "b", "text_hash": "same"},
              {"chunk_id": "c", "text_hash": "other"}]
    out = dedupe_fused(fused, chunks)
    assert [i for i, _ in out] == [0, 2]


def test_institution_filter():
    f = QueryFilters(entities=frozenset({"nbfc"}))
    mask = f.mask(CHUNKS)
    # broad (current) passes; commercial-banks-only fails; withdrawn is dropped
    # by default; market_ops dropped by regulatory_only.
    assert mask.tolist() == [True, False, False, False]


def test_theme_and_since_filter():
    f = QueryFilters(themes=frozenset({"KYC / AML"}))
    assert f.mask(CHUNKS).tolist() == [True, False, False, False]
    f = QueryFilters(since="2025-06-01")
    assert f.mask(CHUNKS).tolist() == [True, False, False, False]


def test_withdrawn_toggle():
    f = QueryFilters(include_withdrawn=True, regulatory_only=False)
    mask = f.mask(CHUNKS)
    assert mask is None or mask.all()  # nothing to filter out -> no mask / all pass


def test_as_of_excludes_not_yet_in_force():
    chunks = [
        {"chunk_id": "old::0000", "doc_id": "old", "content_class": "regulation",
         "status": "current", "effective_from": "2020-01-01",
         "entities": [], "entity_scope": "broad"},
        {"chunk_id": "new::0000", "doc_id": "new", "content_class": "regulation",
         "status": "forthcoming", "effective_from": "2027-04-01",
         "entities": [], "entity_scope": "broad"},
    ]
    assert QueryFilters(as_of="2026-10-20").mask(chunks).tolist() == [True, False]
    assert QueryFilters(as_of="2027-04-01").mask(chunks).tolist() == [True, True]


def test_strict_as_of_excludes_undated_versions():
    chunks = [
        {"chunk_id": "u::0000", "doc_id": "u", "content_class": "regulation",
         "status": "current", "entities": [], "entity_scope": "broad"},
    ]
    assert QueryFilters(as_of="2026-10-20").mask(chunks).tolist() == [True]
    assert QueryFilters(as_of="2026-10-20", strict_as_of=True).mask(chunks).tolist() == [False]


def test_as_of_keeps_superseded_version_in_force():
    chunks = [
        {"chunk_id": "d::0000", "doc_id": "d", "content_class": "regulation",
         "status": "superseded", "superseded_by": "d-r20260101",
         "effective_from": "2020-01-01", "entities": [], "entity_scope": "broad"},
        {"chunk_id": "d-r20260101::0000", "doc_id": "d-r20260101",
         "content_class": "regulation", "status": "current",
         "effective_from": "2026-01-01", "entities": [], "entity_scope": "broad"},
    ]
    # Before the successor takes effect the old version is the law.
    assert QueryFilters(as_of="2025-06-30").mask(chunks).tolist() == [True, False]
    # On/after it, the old version is out and the new one is in.
    assert QueryFilters(as_of="2026-06-30").mask(chunks).tolist() == [False, True]


def test_strict_as_of_drops_later_text():
    chunks = [
        {"chunk_id": "x::0000", "doc_id": "x", "content_class": "regulation",
         "status": "current", "effective_from": "2020-01-01",
         "last_updated": "2026-09-01", "entities": [], "entity_scope": "broad"},
    ]
    assert QueryFilters(as_of="2025-01-01").mask(chunks).tolist() == [True]
    assert QueryFilters(as_of="2025-01-01", strict_as_of=True).mask(chunks).tolist() == [False]


def test_abstain_decision_threshold():
    hit = SimpleNamespace(scores={"rerank": -4.0})
    assert abstain_decision([hit], True, -6.0) is False  # above cutoff -> answer
    assert abstain_decision([hit], True, -2.0) is True  # below cutoff -> refuse
    assert abstain_decision([hit], False, -2.0) is False  # rerank off -> trust hits
    assert abstain_decision([], True, -6.0) is True  # nothing retrieved -> refuse


def test_hit_status_is_derived_at_query_time():
    from regintel.index.fusion import _hit_from_chunk
    chunk = {"chunk_id": "d::0000", "doc_id": "d", "text": "t", "title": "X Amendment Directions, 2026",
             "status": "forthcoming", "effective_from": "2027-04-01"}
    assert _hit_from_chunk(0, chunk, {}, today="2027-03-31").status == "forthcoming"
    h = _hit_from_chunk(0, chunk, {}, today="2027-04-01")        # stored value is stale by now
    assert h.status == "current" and h.effective_from == "2027-04-01"
    chunk["status"] = "superseded"
    assert _hit_from_chunk(0, chunk, {}, today="2030-01-01").status == "superseded"
