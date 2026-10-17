"""Query audit trail — every question, its filters, retrieved passages and answer
settings are appended to data/state/audit.jsonl (masked: no user identity)."""
from __future__ import annotations

from regintel import config, store, util


def append_query(*, session: str, query: str, filters: dict, k: int, rerank: bool,
                 trace: dict, hits: list, answer: dict, mode: str) -> dict:
    record = {
        "ts": util.iso_now(),
        "session": session,
        "query": query,
        "filters": filters,
        "k": k,
        "rerank": rerank,
        "mode": mode,
        "trace": trace,
        "hits": [{
            "ref": f"C{i}",
            "chunk_id": h.chunk_id,
            "doc_id": h.doc_id,
            "reference": h.reference,
            "title": h.title[:120],
            "para": h.para,
            "page": h.page_start,
            "scores": h.scores,
        } for i, h in enumerate(hits, start=1)],
        "answer": {
            "mode": answer.get("mode"),
            "abstain": answer.get("abstain"),
            "model": answer.get("model"),
            "provider": answer.get("provider"),
            "cached": answer.get("cached"),
            "latency_s": answer.get("latency_s"),
            "claims": len(answer.get("claims") or []),
            "claims_dropped": answer.get("claims_dropped"),
            "errors": answer.get("errors"),
        },
    }
    store.append_jsonl(config.AUDIT_LOG, [record])
    return record


def load_recent(limit: int = 200) -> list[dict]:
    return list(reversed(store.load_jsonl(config.AUDIT_LOG)))[:limit]
