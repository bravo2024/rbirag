"""LangChain Embeddings wrapper around the RegIntel bge encoder.

LangChain components (FAISS store, tools, retrievers) use the exact same
vectors as the engine — query prefix included for queries, raw passages for
documents — so neighbours agree between the LangChain path and the hybrid
engine.
"""
from __future__ import annotations

from langchain_core.embeddings import Embeddings

from regintel.index.dense import Encoder


class RegIntelEmbeddings(Embeddings):
    """Wraps `regintel.index.dense.Encoder` (lazy; the model loads on first use)."""

    def __init__(self, model: str | None = None, device: str | None = None):
        self._enc = Encoder(model=model, device=device)

    @property
    def model_name(self) -> str:
        return self._enc.model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._enc.encode_passages(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        return self._enc.encode_queries([text])[0].tolist()
