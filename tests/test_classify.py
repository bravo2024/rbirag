"""Entity, doc-type and theme classification."""
from regintel import classify


def test_entity_from_title():
    cases = [
        ("Reserve Bank of India (Non-Banking Financial Companies - Miscellaneous) Directions, 2025", "nbfc"),
        ("Reserve Bank of India (Housing Finance Companies) Directions, 2025", "hfc"),
        ("Reserve Bank of India (Commercial Banks - Credit Valuation Adjustment Framework) Directions, 2026", "commercial_banks"),
        ("Reserve Bank of India (Mortgage Guarantee Companies) Directions, 2025", "mgc"),
        ("Reserve Bank of India (Payments Banks) Directions, 2025", "payments_banks"),
    ]
    for title, canon in cases:
        entities, evidence, scope = classify.classify_entities(title, "")
        assert canon in entities, (title, entities)
        assert scope == "specific"
        assert evidence and evidence[0][2] == "title"


def test_entity_from_text_head_when_title_generic():
    entities, _, scope = classify.classify_entities(
        "Risk Management and Inter-Bank Dealings",
        "RBI/2026-27/293 To, All Authorised Dealers Madam/Sir, attention is invited...")
    assert entities == ["authorised_dealers"]
    assert scope == "specific"


def test_broad_scope():
    entities, evidence, scope = classify.classify_entities(
        "Reserve Bank of India (All Regulated Entities) Some Framework", "text")
    assert entities == []
    assert scope == "broad"


def test_doc_type():
    assert classify.classify_doc_type(
        "Reserve Bank of India (Commercial Banks - Miscellaneous) Amendment Directions, 2026", None) == "amendment_direction"
    assert classify.classify_doc_type(
        "Reserve Bank of India (Non-Banking Financial Companies) Directions, 2025", None) == "master_direction"
    assert classify.classify_doc_type("A.P. (DIR Series) Circular No. 27", "RBI/2026-27/293") == "circular"
    assert classify.classify_doc_type("Some generic notice", "RBI/2026-27/9") == "notification"


def test_themes_and_content_class():
    themes = classify.classify_themes(
        "Master Direction - Know Your Customer", "The entities shall implement the AML/CFT framework.")
    assert "KYC / AML" in themes
    assert classify.content_class("Auction of Government of India Treasury Bills") == "market_ops"
    assert classify.content_class("Master Direction - KYC") == "regulation"


def test_classify_document_bundle():
    out = classify.classify_document(
        "Reserve Bank of India (Non-Banking Financial Companies - Digital Lending) Directions, 2025",
        "applicable to all Non-Banking Financial Companies", "RBI/2025-26/100")
    assert out["doc_type"] == "master_direction"
    assert "nbfc" in out["entities"]
    assert "Digital Lending / Fintech" in out["themes"]
    assert out["content_class"] == "regulation"


def test_effective_date_whole_document_only():
    # enumerators ("2.", "(ii)") are stripped before the sentence is matched
    assert classify.effective_date(
        "2. These Directions shall come into force with effect from April 1, 2027."
    ) == "2027-04-01"
    assert classify.effective_date(
        "(ii) These Amendment Directions shall come into effect from April 1, 2027."
    ) == "2027-04-01"
    assert classify.effective_date(
        "The Directions shall be applicable to loans sanctioned under the KCC Scheme "
        "with effect from January 01, 2027."
    ) == "2027-01-01"
    # paragraph/section-level commencements do not count
    assert classify.effective_date(
        "(3) Section B of the Directions on implementation of Unique Transaction "
        "Identifier (UTI) shall come into effect from January 01, 2027.") is None
    # the latest whole-document date wins
    assert classify.effective_date(
        "These Directions shall come into force from April 1, 2020. "
        "These Directions shall come into effect from April 1, 2027.") == "2027-04-01"
    assert classify.effective_date("no commencement here") is None


def test_status_for():
    today = "2026-10-17"
    assert classify.status_for("Withdrawn circular on X", "2027-01-01", today=today) == "withdrawn"
    assert classify.status_for(
        "Reserve Bank of India (Commercial Banks - Forthcoming Instructions) Directions, 2025",
        None, today=today) == "forthcoming"
    # an amendment notice acting in 2027 is forthcoming via its date
    assert classify.status_for(
        "Reserve Bank of India (Commercial Banks - Miscellaneous) Amendment Directions, 2026",
        "2027-04-01", today=today) == "forthcoming"
    assert classify.status_for("Some Directions, 2025", "2026-04-01", today=today) == "current"
    assert classify.status_for("Some Directions, 2025", None, today=today) == "current"


def test_effective_status_flips_forthcoming_to_current_on_its_date():
    title = "Reserve Bank of India (Commercial Banks - Miscellaneous) Amendment Directions, 2026"
    assert classify.effective_status("forthcoming", title, "2027-04-01", "2027-03-31") == "forthcoming"
    assert classify.effective_status("forthcoming", title, "2027-04-01", "2027-04-01") == "current"
    assert classify.effective_status("forthcoming", title, "2027-04-01", "2027-09-01") == "current"
    # a stale "current" never hides a not-yet-effective date either
    assert classify.effective_status("current", title, "2027-04-01", "2026-10-17") == "forthcoming"
    assert classify.effective_status(None, title, None, "2026-10-17") == "current"


def test_effective_status_keeps_title_forthcoming_and_lifecycle_states():
    fwd = "Reserve Bank of India (Commercial Banks - Forthcoming Instructions) Directions, 2025"
    assert classify.effective_status("forthcoming", fwd, None, "2030-01-01") == "forthcoming"
    for state in ("withdrawn", "superseded"):    # not date-derived: kept as stored
        assert classify.effective_status(state, "Some Directions", "2020-01-01", "2030-01-01") == state
