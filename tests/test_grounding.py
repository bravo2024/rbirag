"""Numeric grounding: claims whose numbers are absent from the cited text."""
from __future__ import annotations

from regintel.answer import annotate_grounding, check_grounding


def test_number_present():
    assert check_grounding("Maintain CRAR of 9 per cent",
                           ["Bank shall maintain CRAR of 9% at all times"]) == []


def test_number_absent():
    issues = check_grounding("Maintain CRAR of 12 per cent",
                             ["Bank shall maintain CRAR of 9% at all times"])
    assert issues == ["12"]


def test_comma_normalised():
    assert check_grounding("Exposure of Rs 1,00,000",
                           ["exposure limit of Rs. 100000 per borrower"]) == []


def test_annotate_grounding_marks_claims():
    claims = [{"text": "Limit is 90 days", "citations": ["C1"]},
              {"text": "Limit is 45 days", "citations": ["C1"]}]
    n = annotate_grounding(claims, {"C1": "The limit is 90 days from the date of sanction."})
    assert n == 1
    assert claims[0].get("grounded") is True
    assert claims[1].get("number_issues") == ["45"]
