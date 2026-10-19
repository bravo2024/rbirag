"""Answer-layer JSON parsing, citation validation and the no-hits path."""
from regintel import answer


def test_extract_json_with_fences_and_prose():
    text = ('Here you go:\n```json\n{"abstain": false, "answer": "A", '
            '"claims": [{"text": "t", "citations": ["C1"]}]}\n```\nDone.')
    parsed = answer.extract_json(text)
    assert parsed and parsed["answer"] == "A"


def test_extract_json_nested_braces():
    text = '{"answer": "use section {x} as {y}", "claims": []}'
    assert answer.extract_json(text)["answer"].startswith("use section")


def test_extract_json_invalid():
    assert answer.extract_json("no json at all") is None
    assert answer.extract_json('{"broken": ') is None
    assert answer.extract_json('{"abstain": false, "answer": "cut off mid') is None


def test_answer_retries_truncated_json(monkeypatch):
    """A first reply truncated mid-JSON (token budget exhausted) is retried
    with a bigger budget instead of dropping to extractive mode."""
    from regintel import llm as llm_mod
    from regintel.index.fusion import Hit

    hit = Hit(chunk_id="c1", doc_id="d1", text="KYC is updated every two years.",
              title="T", reference="R", issue_date="2025-01-01",
              doc_type="master_direction", status="current", entities=["nbfc"],
              entity_scope="specific", themes=[], section="", para="",
              page_start=1, page_end=1, page_url=None, pdf_url=None,
              scores={"rerank": 5.0})
    replies = [
        '{"abstain": false, "answer": "cut mid',
        '{"abstain": false, "answer": "A grounded answer.", '
        '"claims": [{"text": "KYC every two years", "citations": ["C1"]}]}',
    ]
    budgets = []

    def fake_chat(messages, temperature=0.2, max_tokens=1400, providers=None, pin=None,
                  strict_pin=False):
        budgets.append(max_tokens)
        return {"text": replies[len(budgets) - 1], "model": "m", "provider": "p",
                "cached": False, "attempts": []}

    monkeypatch.setattr(llm_mod, "chat", fake_chat)
    out = answer.answer("how often is KYC updated?", [hit], llm_pin=("p", "m"))
    assert out["mode"] == "llm" and out["parse_retries"] == 1
    assert [c["citations"] for c in out["claims"]] == [["C1"]]
    assert budgets == [2600, 5200]


def test_format_para_ranges():
    assert answer.format_para("4.2", "4.4") == "Paras 4.2\u20134.4"
    assert answer.format_para("4.2") == "Para 4.2"
    assert answer.format_para("4.2", "4.2") == "Para 4.2"
    assert answer.format_para("16", "16.4") == "Para 16"   # parent covers sub-paras
    assert answer.format_para("4.2", "(b)") == "Para 4.2"  # mixed label styles
    assert answer.format_para("", "4.4") is None


def test_build_context_uses_para_range():
    from regintel.index.fusion import Hit
    hit = Hit(chunk_id="c1", doc_id="d1", text="The rule text.", title="T",
              reference="R", issue_date="2025-01-01", doc_type="master_direction",
              status="current", entities=["nbfc"], entity_scope="specific",
              themes=[], section="Chapter II", para="4.2", para_end="4.4",
              page_start=3, page_end=3, page_url=None, pdf_url=None, scores={})
    assert "Paras 4.2\u20134.4" in answer.build_context([hit])


def test_validate_claims():
    claims = [
        {"text": "valid claim", "citations": ["C1", "C3"]},
        {"text": "out of range", "citations": ["C99"]},
        {"text": "no citations", "citations": []},
        {"text": "mixed", "citations": ["C2", "C404"]},
        "not-a-dict",
    ]
    valid, dropped = answer.validate_claims(claims, n_passages=5)
    assert [c["text"] for c in valid] == ["valid claim", "mixed"]
    assert valid[1]["citations"] == ["C2"]
    assert dropped == 3


def test_answer_without_hits():
    out = answer.answer("anything", hits=[], use_llm=False)
    assert out["mode"] == "abstain" and out["abstain"] is True
    assert "No passage" in out["answer_md"]


def _hit(text: str = "KYC is updated every two years."):
    from regintel.index.fusion import Hit
    return Hit(chunk_id="c1", doc_id="d1", text=text, title="T", reference="R",
               issue_date="2025-01-01", doc_type="master_direction", status="current",
               entities=["nbfc"], entity_scope="specific", themes=[], section="", para="",
               page_start=1, page_end=1, page_url=None, pdf_url=None,
               scores={"rerank": 5.0})


def test_refuse_shape():
    """Retrieval-level refusal: no LLM, threshold visible, passages kept."""
    out = answer.refuse("fed funds rate", [_hit()], threshold=2.0)
    assert out["mode"] == "abstain" and out["abstain"] is True
    assert "2.0" in out["answer_md"] and out["citations"]
    assert out["citations"][0]["ref"] == "C1"
    assert out["model"] is None and out["latency_s"] == 0.0


def test_low_confidence_note_reaches_the_prompt(monkeypatch):
    from regintel import llm as llm_mod

    seen = {}

    def fake_chat(messages, temperature=0.2, max_tokens=1400, providers=None, pin=None,
                  strict_pin=False):
        seen["user"] = messages[1]["content"]
        return {"text": '{"abstain": true, "answer": "", "claims": []}',
                "model": "m", "provider": "p", "cached": False, "attempts": []}

    monkeypatch.setattr(llm_mod, "chat", fake_chat)
    out = answer.answer("q", [_hit()], low_confidence=True, llm_pin=("p", "m"))
    assert "LOW confidence" in seen["user"] and '"abstain": true' in seen["user"]
    assert out["mode"] == "abstain"

    out2 = answer.answer("q", [_hit()], low_confidence=False, llm_pin=("p", "m"))
    assert "LOW confidence" not in seen["user"]
    assert out2["low_confidence"] is False


def _scoped_hit(i, scope, ents, text="Rule text."):
    from regintel.index.fusion import Hit
    return Hit(chunk_id=f"c{i}", doc_id=f"d{i}", text=text, title=f"Title {i}", reference=None,
               issue_date=None, doc_type="master_direction", status="current", entities=ents,
               entity_scope=scope, themes=[], section="", para="", page_start=0, page_end=0,
               page_url=None, pdf_url=None, scores={})


def test_build_context_header_names_who_each_passage_applies_to():
    ctx = answer.build_context([
        _scoped_hit(1, "specific", ["commercial_banks"]),
        _scoped_hit(2, "specific", ["cooperative_banks", "regional_rural_banks"]),
        _scoped_hit(3, "broad", []),
    ])
    c1, c2, c3 = ctx.split("\n\n")
    assert "applies to: Commercial Banks" in c1.splitlines()[0]
    assert "applies to: Co-operative Banks, Regional Rural Banks" in c2.splitlines()[0]
    assert "applies to: all regulated entities" in c3.splitlines()[0]


def test_system_prompt_forbids_merging_entity_versions():
    assert "applies to" in answer.SYSTEM_PROMPT
    assert "never merge" in answer.SYSTEM_PROMPT


def test_entity_groups_counts_distinct_specific_scopes_only():
    hits = [_scoped_hit(1, "specific", ["commercial_banks"]), _scoped_hit(2, "specific", ["commercial_banks"]),
            _scoped_hit(3, "broad", []), _scoped_hit(4, "specific", ["nbfc", "hfc"])]
    assert answer.entity_groups(hits) == [("commercial_banks",), ("hfc", "nbfc")]
    cards = [{"entity_scope": "specific", "entities": ["nbfc"]}, {"entity_scope": "broad"}]
    assert answer.entity_groups(cards) == [("nbfc",)]       # works on citation cards too
    assert answer.entity_groups([_scoped_hit(5, "broad", [])]) == []
