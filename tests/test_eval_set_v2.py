"""Integrity checks for the draft 120-question v2 eval set."""
from __future__ import annotations

import json
from pathlib import Path

from regintel import store

ROOT = Path(__file__).resolve().parents[1]
SET = ROOT / "evals" / "eval_set_v2.jsonl"
CATEGORIES = {"answerable": 60, "as_of": 20, "amendment": 20, "unanswerable": 20}
DOCS = store.load_documents()


def _rows() -> list[dict]:
    return [json.loads(line) for line in SET.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def test_counts_and_unique_ids():
    rows = _rows()
    assert len(rows) == 120
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    assert counts == CATEGORIES
    ids = [row["id"] for row in rows]
    assert len(ids) == len(set(ids))


def test_rows_are_reviewable_drafts():
    for row in _rows():
        assert row["review_status"] == "draft"
        assert row["query"].strip()
        assert row["category"] in CATEGORIES
        assert "reviewer_notes" in row
        if row["category"] in ("answerable", "amendment"):
            assert row["gold_doc_ids"], row["id"]
            assert row["expected"] == "answer"
        elif row["category"] == "as_of":
            assert row["as_of"]
            assert isinstance(row["expect_present"], bool)
            assert row["probe_doc_id"] in row["gold_doc_ids"]
            assert row["expected"] in ("included", "excluded")
        else:
            assert row["gold_doc_ids"] == []
            assert row["expected"] == "abstain"


def test_gold_documents_exist():
    for row in _rows():
        for doc_id in row["gold_doc_ids"]:
            assert doc_id in DOCS, f"{row['id']} references missing {doc_id}"


def test_as_of_expectations_match_effective_dates():
    for row in _rows():
        if row["category"] != "as_of":
            continue
        probe = DOCS[row["probe_doc_id"]]
        effective = str(probe.get("effective_from") or "")[:10]
        assert effective, f"{row['id']} probe has no effective date"
        in_force = effective <= row["as_of"]
        assert in_force == row["expect_present"], (
            f"{row['id']}: effective={effective} as_of={row['as_of']} "
            f"expected_present={row['expect_present']}")
