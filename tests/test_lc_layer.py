"""LangChain layer: retriever documents, LCEL chain end-to-end offline."""
from __future__ import annotations

import json

import pytest

pytest.importorskip("langchain_core")

from langchain_core.language_models.fake_chat_models import FakeListChatModel  # noqa: E402

from regintel.index.fusion import Hit  # noqa: E402
from regintel.lc.chain import build_answer_chain  # noqa: E402
from regintel.lc.retriever import RegIntelRetriever  # noqa: E402


def _hit(text: str = "The limit is 90 days from sanction.") -> Hit:
    return Hit(chunk_id="md1::0001", doc_id="md1", text=text, title="Master Direction - Test",
               reference="RBI/2026-27/1", issue_date="2026-04-01", doc_type="master_direction",
               status="current", entities=["nbfc"], entity_scope="specific", themes=["kyc"],
               section="Chapter II", para="4.2", page_start=3, page_end=4,
               page_url="https://rbi.org.in/x", pdf_url="https://rbidocs.rbi.org.in/x.pdf",
               scores={"rerank": 1.2})


class _FakeEngine:
    def __init__(self, hits):
        self._hits = hits

    def retrieve(self, query, filters=None, k=None, rerank=None, candidates=None, debug=False):
        return {"hits": self._hits[: (k or 6)], "abstain": False, "trace": {"n_chunks": 1}}


def test_retriever_maps_hits_to_documents():
    r = RegIntelRetriever(engine=_FakeEngine([_hit()]))
    docs = r.invoke("what is the limit")
    assert len(docs) == 1
    d = docs[0]
    assert d.metadata["chunk_id"] == "md1::0001"
    assert d.metadata["para"] == "4.2"
    assert "90 days" in d.page_content


def test_chain_validates_and_grounds_claims():
    reply = json.dumps({"abstain": False, "answer": "Ninety days applies.",
                        "claims": [{"text": "The limit is 90 days", "citations": ["C1"]}]})
    engine = _FakeEngine([_hit()])
    chain = build_answer_chain(engine=engine, llm_override=FakeListChatModel(responses=[reply]))
    out = chain.invoke("limit?")
    assert out["mode"] == "llm"
    assert out["claims"][0]["citations"] == ["C1"]
    assert out["claims"][0].get("grounded") is True
    assert out["citations"][0]["ref"] == "C1"
    assert out["engine"] == "langchain"


def test_chain_flags_ungrounded_number():
    reply = json.dumps({"abstain": False, "answer": "x",
                        "claims": [{"text": "The limit is 45 days", "citations": ["C1"]}]})
    engine = _FakeEngine([_hit()])
    chain = build_answer_chain(engine=engine, llm_override=FakeListChatModel(responses=[reply]))
    out = chain.invoke("limit?")
    assert out["claims_ungrounded"] == 1
    assert out["claims"][0]["number_issues"] == ["45"]


def test_chain_drops_unresolvable_citation():
    reply = json.dumps({"abstain": False, "answer": "x",
                        "claims": [{"text": "made up", "citations": ["C9"]}]})
    engine = _FakeEngine([_hit()])
    chain = build_answer_chain(engine=engine, llm_override=FakeListChatModel(responses=[reply]))
    out = chain.invoke("q")
    assert out["claims"] == [] and out["claims_dropped"] == 1


def test_chain_extractive_without_llm():
    engine = _FakeEngine([_hit()])
    chain = build_answer_chain(engine=engine, use_llm=False)
    out = chain.invoke("q")
    assert out["mode"] == "extractive"
    assert "[C1]" in out["answer_md"]
