"""LCEL answer chain: hybrid retrieval -> prompt -> failover chat model ->
the same strict-JSON citation validator the native path uses.

This is the LangChain-native route through the product: it calls the full
retrieval engine (BM25 + dense + RRF + cross-encoder rerank, not a plain
vector lookup), the portfolio's provider failover, and regintel.answer's
validator + numeric-grounding checks, so the output contract is identical to
`regintel.answer.answer()` and directly comparable.

    from regintel.lc import lc_answer
    lc_answer(engine, "What is the CRAR requirement for NBFCs?")
"""
from __future__ import annotations

import time
from typing import Any, Optional

from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

from regintel import answer as answer_mod
from regintel import config
from regintel.lc.retriever import RegIntelRetriever

USER_TEMPLATE = ("QUESTION: {question}\n\nCONTEXT:\n{context}\n\n"
                 "Answer with strict JSON per the system instructions.")


def format_docs(docs) -> str:
    """Number the retrieved passages [C1]…[Cn] exactly like the native path."""
    parts = []
    for i, d in enumerate(docs, start=1):
        m = d.metadata
        meta = " | ".join(x for x in [
            m.get("title"), m.get("reference"),
            f"dated {m['issue_date']}" if m.get("issue_date") else None,
            m.get("section") or None, f"Para {m['para']}" if m.get("para") else None,
            f"pp. {m['page_start']}-{m['page_end']}" if m.get("page_start") else None,
            f"[{m['status']}]" if m.get("status") not in (None, "current") else None] if x)
        parts.append(f"[C{i}] {meta}\n{d.page_content}")
    return "\n\n".join(parts)


def cards_from_docs(docs) -> list[dict]:
    """Citation cards (same fields the native path resolves from Hit objects)."""
    cards = []
    for i, d in enumerate(docs, start=1):
        m = d.metadata
        cards.append({"ref": f"C{i}", "chunk_id": m.get("chunk_id"), "title": m.get("title"),
                      "reference": m.get("reference"), "issue_date": m.get("issue_date"),
                      "doc_type": m.get("doc_type"), "status": m.get("status"),
                      "entities": m.get("entities"), "entity_scope": m.get("entity_scope"),
                      "section": m.get("section"), "para": m.get("para"),
                      "page_start": m.get("page_start"), "page_end": m.get("page_end"),
                      "page_url": m.get("page_url"), "pdf_url": m.get("pdf_url"),
                      "scores": m.get("scores") or {}, "snippet": d.page_content[:420]})
    return cards


def _finalize(state: dict) -> dict:
    query = state["question"]
    docs = state.get("docs") or []
    raw = state.get("raw") or ""
    t0 = state.get("_t0", time.time())
    cards = cards_from_docs(docs)
    result = {
        "query": query, "mode": "abstain", "abstain": True, "low_confidence": False,
        "answer_md": "", "claims": [], "claims_dropped": 0, "claims_ungrounded": 0,
        "citations": cards, "model": state.get("_model"), "provider": None, "cached": False,
        "latency_s": 0.0, "errors": [], "engine": "langchain",
    }
    if not docs:
        result["answer_md"] = ("No passage in the corpus (under the current filters) covers this "
                               "question. Try widening the institution filter or syncing new "
                               "notifications first.")
        result["latency_s"] = round(time.time() - t0, 2)
        return result
    try:
        parsed = answer_mod.extract_json(raw)
        if parsed is None:
            raise ValueError("reply is not valid JSON")
        claims, dropped = answer_mod.validate_claims(parsed.get("claims") or [], len(docs))
        ungrounded = answer_mod.annotate_grounding(
            claims, {f"C{i}": d.page_content for i, d in enumerate(docs, start=1)})
        answer_md = (parsed.get("answer") or "").strip()
        abstain = bool(parsed.get("abstain")) or (not claims and not answer_md)
        if not answer_md and abstain:
            answer_md = ("The retrieved passages do not contain a direct answer. Closest sources "
                         "are listed below in the audit trail.")
        result.update(mode="abstain" if abstain else "llm", abstain=abstain, answer_md=answer_md,
                      claims=claims, claims_dropped=dropped, claims_ungrounded=ungrounded)
    except Exception as exc:  # noqa: BLE001
        result.update(mode="extractive", abstain=False,
                      answer_md=answer_mod._extractive(query, cards),
                      errors=[f"{type(exc).__name__}: {str(exc)[:200]}"])
    result["latency_s"] = round(time.time() - t0, 2)
    return result


def build_answer_chain(engine: Any = None, *, retriever: Any = None, filters: Any = None,
                       top_k: Optional[int] = None, rerank: bool = True,
                       temperature: float = 0.2, max_tokens: int = 1400,
                       llm_override: Any = None, use_llm: bool = True,
                       model_label: Optional[str] = None) -> Runnable:
    """Compose the LCEL chain.

    `retriever` overrides the default RegIntelRetriever (used by tests);
    `llm_override` (any chat-model runnable) keeps tests offline;
    `use_llm=False` returns the extractive shape without any model.
    """
    retriever = retriever or RegIntelRetriever(
        engine=engine, filters=filters, top_k=top_k or config.TOP_K, rerank=rerank)
    if not use_llm:
        model: Any = RunnableLambda(lambda _: "")
        label = "extractive"
    elif llm_override is not None:
        model = llm_override
        label = (model_label or getattr(llm_override, "model_name", None)
                 or type(llm_override).__name__)
    else:
        from regintel.lc.llm import chat_model  # heavy import, only when needed

        model = chat_model(temperature=temperature, max_tokens=max_tokens)
        label = model_label or getattr(model, "model_name", None)
    # the system prompt is a constant (it contains literal JSON braces), so it
    # is passed as a SystemMessage — only the human template is templated
    prompt = ChatPromptTemplate.from_messages([SystemMessage(content=answer_mod.SYSTEM_PROMPT),
                                               ("human", USER_TEMPLATE)])
    messages_from_state = RunnableLambda(
        lambda s: prompt.format_messages(question=s["question"], context=format_docs(s["docs"])))
    # accept either a question string or a state dict; assign() needs a dict
    normalize = RunnableLambda(lambda x: {"question": x} if isinstance(x, str) else x)
    chain = (
        normalize
        | RunnablePassthrough.assign(
            docs=RunnableLambda(lambda s: retriever.invoke(s["question"])),
            _t0=RunnableLambda(lambda _s: time.time()),
            _model=RunnableLambda(lambda _s: label),
        )
        .assign(raw=messages_from_state | model | StrOutputParser())
        .assign(answer=RunnableLambda(_finalize))
        | RunnableLambda(lambda s: s["answer"])
    )
    return chain


def lc_answer(engine: Any, query: str, *, use_llm: bool = True, **kw) -> dict:
    """One-shot convenience: same output contract as regintel.answer.answer()."""
    return build_answer_chain(engine, use_llm=use_llm, **kw).invoke(query)
