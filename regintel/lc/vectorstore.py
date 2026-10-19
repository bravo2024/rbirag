"""Persisted LangChain FAISS store over the same vectors the engine searches.

Built from data/chunks.jsonl + data/index/dense_fp16.npy, saved to
data/index/lc_faiss/ (index.faiss + docstore). It is the LangChain-native
handle on the corpus — similarity_search, MMR, as_retriever() — while the
hybrid engine remains the primary retrieval path; both read the same bge
vectors, so the dense neighbours agree.

    python scripts/build_lc_store.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from regintel import config, store
from regintel.index.dense import DenseIndex

META_KEYS = ("chunk_id", "doc_id", "title", "reference", "issue_date", "as_of", "doc_type",
             "status", "entities", "entity_scope", "themes", "section", "para", "para_end",
             "page_start", "page_end", "page_url", "pdf_url")


def default_dir() -> Path:
    return config.INDEX_DIR / "lc_faiss"


def _embeddings():
    from regintel.lc.embeddings import RegIntelEmbeddings

    return RegIntelEmbeddings()


def build_vector_store(persist_dir: Optional[Path] = None, embeddings=None, progress=print):
    """(Re)build the store from the committed artifacts; returns the FAISS store."""
    from langchain_community.vectorstores import FAISS
    from langchain_community.vectorstores.utils import DistanceStrategy

    chunks = store.load_chunks()
    dense = DenseIndex.load(config.INDEX_DIR)
    if dense.n != len(chunks):
        raise RuntimeError(f"dense matrix has {dense.n} vectors for {len(chunks)} chunks — "
                           "rebuild the index first (scripts/build_index.py)")
    texts = [c["text"] for c in chunks]
    vectors = dense.matrix.astype("float32").tolist()
    metadatas = [{k: c.get(k) for k in META_KEYS} for c in chunks]
    vs = FAISS.from_embeddings(
        list(zip(texts, vectors)), embeddings or _embeddings(), metadatas=metadatas,
        ids=[c["chunk_id"] for c in chunks],
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
    out = Path(persist_dir or default_dir())
    out.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(out))
    progress(f"langchain faiss store: {dense.n} vectors -> {out}")
    return vs


def load_vector_store(persist_dir: Optional[Path] = None):
    """Load the persisted store (queries embed through the bge encoder)."""
    from langchain_community.vectorstores import FAISS

    out = Path(persist_dir or default_dir())
    return FAISS.load_local(str(out), _embeddings(), allow_dangerous_deserialization=True)
