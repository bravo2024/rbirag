"""LangChain layer for RBIRAG.

The hybrid engine stays the retrieval core; this package exposes it to the
LangChain ecosystem (retriever, LCEL answer chain, persisted FAISS store,
failover chat models) so the same machinery can be composed into chains or
agents without reimplementing any domain logic.

Imports are lazy: `import regintel.lc` pulls only langchain-core; the FAISS
store and the OpenAI-compatible chat models load on first use.
"""
from __future__ import annotations

from regintel.lc.embeddings import RegIntelEmbeddings
from regintel.lc.retriever import RegIntelRetriever

__all__ = ["RegIntelEmbeddings", "RegIntelRetriever", "build_answer_chain", "lc_answer",
           "build_vector_store", "load_vector_store", "chat_model", "build_chat_models"]


def __getattr__(name: str):
    if name in ("build_answer_chain", "lc_answer"):
        from regintel.lc import chain

        return getattr(chain, name)
    if name in ("build_vector_store", "load_vector_store"):
        from regintel.lc import vectorstore

        return getattr(vectorstore, name)
    if name in ("chat_model", "build_chat_models"):
        from regintel.lc import llm as lc_llm

        return getattr(lc_llm, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
