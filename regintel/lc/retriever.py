"""The RegIntel hybrid engine exposed as a LangChain retriever.

Chains and agents get the full quality stack — BM25 + dense + RRF fusion +
cross-encoder rerank, under institution / doc-type / theme / date filters —
not just a vector lookup. Each returned Document carries the full citation
metadata (chunk id, paragraph, pages, provenance URLs, per-arm scores).
"""
from __future__ import annotations

from typing import Any, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from regintel.index.fusion import Hit


def doc_from_hit(h: Hit) -> Document:
    md = {"chunk_id": h.chunk_id, "doc_id": h.doc_id, "title": h.title, "reference": h.reference,
          "issue_date": h.issue_date, "doc_type": h.doc_type, "status": h.status,
          "entities": h.entities, "entity_scope": h.entity_scope, "themes": h.themes,
          "section": h.section, "para": h.para, "page_start": h.page_start,
          "page_end": h.page_end, "page_url": h.page_url, "pdf_url": h.pdf_url,
          "scores": h.scores}
    return Document(page_content=h.text, metadata=md)


class RegIntelRetriever(BaseRetriever):
    """LangChain retriever over the hybrid engine.

    `filters` is a regintel.index.fusion.QueryFilters (or None for no
    filtering). `top_k`/`rerank` map onto the engine's knobs.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    engine: Any
    filters: Optional[Any] = None
    top_k: int = 6
    rerank: bool = True

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        out = self.engine.retrieve(query, filters=self.filters, k=self.top_k, rerank=self.rerank)
        return [doc_from_hit(h) for h in out["hits"]]
