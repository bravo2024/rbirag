"""Hybrid retrieval engine: BM25 + dense -> RRF fusion -> cross-encoder rerank,
under metadata filters (institution, doc type, theme, date, status).

This is the retrieval core the app and the eval script share.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from regintel import classify, config, store
from regintel.index.bm25 import BM25Index
from regintel.index.dense import DenseIndex, Encoder, check_alignment, dense_meta
from regintel.index.rerank import Reranker

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
def _valid_from(c: dict) -> str | None:
    """First date this version was in force (ISO string, best available)."""
    return c.get("effective_from") or c.get("as_of") or c.get("issue_date")


def _superseded_dates(chunks: list[dict]) -> dict[str, str | None]:
    """doc_id -> date its successor took effect, for superseded versions."""
    marked = {}
    for c in chunks:
        if c.get("doc_id") and c.get("superseded_by"):
            marked[c["doc_id"]] = c
    by_doc = {c.get("doc_id"): c for c in chunks if c.get("doc_id")}
    out: dict[str, str | None] = {}
    for doc_id, c in marked.items():
        nxt = by_doc.get(c["superseded_by"])
        if nxt is not None:
            out[doc_id] = _valid_from(nxt)
    return out


@dataclass
class QueryFilters:
    """Chunk-level filters. `entities` empty = every institution."""
    entities: frozenset[str] = frozenset()
    doc_types: frozenset[str] = frozenset()
    themes: frozenset[str] = frozenset()
    since: str | None = None            # ISO date; undated docs are kept
    include_withdrawn: bool = False
    regulatory_only: bool = True
    as_of: str | None = None            # ISO date; only versions in force then
    strict_as_of: bool = False          # also drop text last updated after as_of

    def mask(self, chunks: list[dict]) -> np.ndarray | None:
        if not (self.entities or self.doc_types or self.themes or self.since
                or self.as_of or self.strict_as_of
                or self.include_withdrawn is False or self.regulatory_only):
            return None
        superseded_on = _superseded_dates(chunks) if self.as_of else {}
        m = np.ones(len(chunks), dtype=bool)
        for i, c in enumerate(chunks):
            if self.regulatory_only and c.get("content_class") != "regulation":
                m[i] = False
                continue
            if self.as_of:
                vf = _valid_from(c)
                if vf and vf > self.as_of:
                    m[i] = False          # not yet in force / published on that date
                    continue
                if self.strict_as_of:
                    if not vf:
                        m[i] = False      # undated: cannot prove it was in force
                        continue
                    text_date = (c.get("last_updated") or c.get("as_of")
                                 or c.get("issue_date"))
                    if text_date and text_date > self.as_of:
                        m[i] = False  # text includes amendments made after the date
                        continue
            if not self.include_withdrawn and c.get("status") in ("withdrawn", "superseded"):
                # A superseded version was still the law before its successor
                # took effect, so an as-of query before that date keeps it.
                if c.get("status") == "superseded" and self.as_of:
                    end = superseded_on.get(c.get("doc_id"))
                    if end and self.as_of < end:
                        pass
                    else:
                        m[i] = False
                        continue
                else:
                    m[i] = False
                    continue
            if self.since:
                d = c.get("as_of") or c.get("issue_date")
                if d and d < self.since:
                    m[i] = False
                    continue
            if self.doc_types and c.get("doc_type") not in self.doc_types:
                m[i] = False
                continue
            if self.themes and not (set(c.get("themes") or []) & set(self.themes)):
                m[i] = False
                continue
            if self.entities:
                if c.get("entity_scope") == "broad":
                    continue
                if not (set(c.get("entities") or []) & set(self.entities)):
                    m[i] = False
        return m


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------
def rrf_fuse(rankings: list[list[int]], k: int = None, weights: list[float] | None = None
             ) -> list[tuple[int, float]]:
    """Reciprocal rank fusion. Each ranking is a list of ids, best first."""
    k = k or config.RRF_K
    weights = weights or [1.0] * len(rankings)
    scores: dict[int, float] = {}
    for w, ranking in zip(weights, rankings):
        for rank, i in enumerate(ranking, start=1):
            scores[i] = scores.get(i, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def dedupe_fused(fused: list[tuple[int, float]], chunks: list[dict]) -> list[tuple[int, float]]:
    """Drop repeated passages (same text_hash, e.g. a para verbatim in both the
    Bank and NBFC version of a Direction); keep the best-ranked copy."""
    seen: set[str] = set()
    out: list[tuple[int, float]] = []
    for i, score in fused:
        h = chunks[i].get("text_hash") or chunks[i]["chunk_id"]
        if h in seen:
            continue
        seen.add(h)
        out.append((i, score))
    return out


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    text: str
    title: str
    reference: str | None
    issue_date: str | None
    doc_type: str
    status: str
    entities: list[str]
    entity_scope: str
    themes: list[str]
    section: str
    para: str
    page_start: int
    page_end: int
    page_url: str | None
    pdf_url: str | None
    scores: dict = field(default_factory=dict)
    effective_from: str | None = None
    amends: list = field(default_factory=list)
    amended_by: list = field(default_factory=list)
    superseded_by: str | None = None
    para_end: str = ""

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        return d


def _hit_from_chunk(i: int, chunk: dict, scores: dict, today: str | None = None) -> Hit:
    # status is re-derived against today: the stored value is a snapshot from
    # ingest / the last recompute, and a forthcoming doc must turn current on
    # its effective date without a recompute
    status = classify.effective_status(chunk.get("status"), chunk.get("title", ""),
                                       chunk.get("effective_from"), today)
    return Hit(
        chunk_id=chunk["chunk_id"], doc_id=chunk["doc_id"], text=chunk["text"],
        title=chunk.get("title", ""), reference=chunk.get("reference"),
        issue_date=chunk.get("issue_date") or chunk.get("as_of"), doc_type=chunk.get("doc_type", "other"),
        status=status, entities=chunk.get("entities") or [],
        entity_scope=chunk.get("entity_scope", "specific"),
        themes=chunk.get("themes") or [], section=chunk.get("section", ""),
        para=chunk.get("para", ""), page_start=chunk.get("page_start", 0),
        page_end=chunk.get("page_end", 0), page_url=chunk.get("page_url"),
        pdf_url=chunk.get("pdf_url"), scores=scores,
        effective_from=chunk.get("effective_from"),
        amends=chunk.get("amends") or [], amended_by=chunk.get("amended_by") or [],
        superseded_by=chunk.get("superseded_by"), para_end=chunk.get("para_end", ""))


def abstain_decision(hits: list[Hit], rerank: bool, threshold: float) -> bool:
    """Retrieval-level abstain: no hits, or the best rerank logit below ``threshold``.

    Without rerank scores there is no calibrated cutoff, so the engine trusts
    the retrieved hits (unchanged behaviour when reranking is off).
    """
    if not hits:
        return True
    if rerank and "rerank" in hits[0].scores:
        return hits[0].scores["rerank"] < threshold
    return False


class Engine:
    """Loads the built artifacts once; serves retrieval for app + evals."""

    def __init__(self, index_dir=None):
        self.index_dir = index_dir or config.INDEX_DIR
        self.chunks: list[dict] = store.load_chunks()
        self.bm25: BM25Index | None = None
        self.dense: DenseIndex | None = None
        # Queries must live in the same embedding space as the stored vectors.
        # dense_meta is authoritative; config.EMBED_MODEL only seeds a new build.
        meta = dense_meta(self.index_dir) or {}
        encoder_name = meta.get("encoder") or config.EMBED_MODEL
        if meta.get("encoder") and meta["encoder"] != config.EMBED_MODEL:
            log.warning("query encoder %r taken from dense_meta (config default is %r)",
                        encoder_name, config.EMBED_MODEL)
        self.encoder = Encoder(model=encoder_name)
        self._reranker: Reranker | None = None
        try:
            self.bm25 = BM25Index.load(self.index_dir)
            self.dense = DenseIndex.load(self.index_dir)
        except FileNotFoundError:
            pass  # index not built yet; UI checks Engine.ready
        if self.dense is not None and self.chunks:
            # raises IndexMismatchError for a stale index; legacy = no fingerprint
            if check_alignment(self.index_dir, self.chunks) == "legacy":
                log.warning("dense index has no alignment fingerprint (built before it existed); "
                            "alignment with chunks.jsonl is unverified — rebuild to add one")

    @property
    def ready(self) -> bool:
        return bool(self.chunks) and self.bm25 is not None and self.dense is not None

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = Reranker()
        return self._reranker

    def stats(self) -> dict:
        manifest = store.read_manifest() or {}
        return {"n_chunks": len(self.chunks),
                "n_docs": len({c["doc_id"] for c in self.chunks}),
                "dense_backend": self.dense.backend if self.dense is not None else None,
                **{k: manifest.get(k) for k in ("version", "built_at")}}

    def retrieve(self, query: str, filters: QueryFilters | None = None,
                 k: int = None, rerank: bool = None, candidates: int = None,
                 abstain_score: float | None = None, debug: bool = False,
                 today: str | None = None) -> dict:
        """Returns {"hits": [Hit], "abstain": bool, "trace": {...}}.

        ``abstain_score`` overrides ``config.ABSTAIN_SCORE`` for this call (the
        app's threshold slider); ``None`` keeps the configured default.
        ``today`` (ISO date) pins the date used to derive each hit's status.
        """
        k = k or config.TOP_K
        rerank = True if rerank is None else rerank
        candidates = candidates or config.CANDIDATES
        threshold = config.ABSTAIN_SCORE if abstain_score is None else abstain_score
        if not self.ready:
            raise RuntimeError("index not built — run scripts/sync.py (or build_index.py) first")

        mask = filters.mask(self.chunks) if filters else None
        if mask is not None and not mask.any():
            return {"hits": [], "abstain": True,
                    "trace": {"reason": "no documents pass the filters", "masked_out": len(self.chunks)}}

        t0 = time.perf_counter()
        bm = self.bm25.search(query, k=candidates, mask=mask)
        t1 = time.perf_counter()
        qvec = self.encoder.encode_queries([query])[0]
        t2 = time.perf_counter()
        dn = self.dense.search(qvec, k=candidates, mask=mask)
        t3 = time.perf_counter()

        bm_order = [i for i, _ in bm]
        dn_order = [i for i, _ in dn]
        fused = dedupe_fused(rrf_fuse([bm_order, dn_order]), self.chunks)
        t4 = time.perf_counter()

        top = fused[: config.RERANK_DEPTH]
        rerank_scores: dict[int, float] = {}
        if rerank and top:
            # Full passage text — the median passage is ~340 words, so a
            # character pre-cut hides the tail from the cross-encoder. Length
            # is bounded by the tokenizer's RERANK_MAX_LENGTH (512 tokens).
            texts = [self.chunks[i]["text"] for i, _ in top]
            for (i, _), s in zip(top, self.reranker.score(query, texts)):
                rerank_scores[i] = s
            top = sorted(top, key=lambda kv: -rerank_scores.get(kv[0], -1e9))[:k]
        else:
            top = top[:k]
        t5 = time.perf_counter()

        # An as-of query derives each hit's status at that date, not today.
        if today is None and filters is not None:
            today = filters.as_of

        bm_rank = {i: r + 1 for r, (i, _) in enumerate(bm)}
        dn_rank = {i: r + 1 for r, (i, _) in enumerate(dn)}
        dn_score = dict(dn)
        hits: list[Hit] = []
        for i, rrf_score in top:
            scores = {"rrf": round(rrf_score, 6),
                      "bm25_rank": bm_rank.get(i),
                      "dense_rank": dn_rank.get(i),
                      "dense_score": round(dn_score.get(i, 0.0), 4)}
            if i in rerank_scores:
                scores["rerank"] = round(rerank_scores[i], 3)
            hits.append(_hit_from_chunk(i, self.chunks[i], scores, today))

        abstain = abstain_decision(hits, rerank, threshold)

        trace = {"n_chunks": len(self.chunks),
                 "as_of": filters.as_of if filters else None,
                 "masked_out": int(len(self.chunks) - mask.sum()) if mask is not None else 0,
                 "abstain_threshold": threshold,
                 "bm25_hits": len(bm), "dense_hits": len(dn),
                 "fused": len(fused), "reranked": len(rerank_scores),
                 "timings_ms": {"bm25": round((t1 - t0) * 1000), "encode": round((t2 - t1) * 1000),
                                "dense": round((t3 - t2) * 1000), "fuse": round((t4 - t3) * 1000),
                                "rerank": round((t5 - t4) * 1000), "total": round((t5 - t0) * 1000)}}
        if debug:
            trace["top_candidates"] = [
                {"chunk_id": self.chunks[i]["chunk_id"], "rrf": round(s, 6)} for i, s in fused[:10]]
        return {"hits": hits, "abstain": abstain, "trace": trace}
