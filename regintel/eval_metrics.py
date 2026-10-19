"""Pure IR metric helpers shared by the eval scripts (no I/O, unit-tested).

Document-level, binary relevance: a retrieved passage counts if its doc_id is
in the query's gold set.
"""
from __future__ import annotations

import math


def first_hit_rank(docs: list[str], gold: set[str], k: int) -> int | None:
    """1-based rank of the first gold doc within top-k, or None."""
    for i, d in enumerate(docs[:k], start=1):
        if d in gold:
            return i
    return None


def hit_at_k(docs: list[str], gold: set[str], k: int) -> bool:
    return first_hit_rank(docs, gold, k) is not None


def mrr_at_k(docs: list[str], gold: set[str], k: int) -> float:
    r = first_hit_rank(docs, gold, k)
    return 1.0 / r if r else 0.0


def ndcg_at_k(docs: list[str], gold: set[str], k: int) -> float:
    """Binary-gain nDCG@k at document level.

    Only the first passage of each gold document earns gain (passages from
    the same document are not double-counted); the ideal ranking grants one
    gain per gold document, so the score stays within [0, 1].
    """
    dcg, seen = 0.0, set()
    for i, d in enumerate(docs[:k], start=1):
        if d in gold and d not in seen:
            seen.add(d)
            dcg += 1.0 / math.log2(i + 1)
    ideal = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal + 1))
    return dcg / idcg if idcg else 0.0


def pearson(xs: list[float], ys: list[float]) -> float | None:
    """Pearson correlation; None when undefined (fewer than 2 points or a
    zero-variance side)."""
    n = len(xs)
    if n < 2 or n != len(ys):
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return None
    return cov / math.sqrt(vx * vy)


def judge_agreement(scores1: list, served1: list[dict], scores2: list, served2: list[dict]) -> dict:
    """Agreement between two judges, over records where they were different models.

    ``served*`` hold, per record, the {"provider/model": calls} census of the
    judge calls behind that score. A record counts only when both scores exist
    and the two judges' served-model sets are disjoint: when a fallback (or a
    cross-model cache hit) made the same model serve both judges, the pair is
    one judge compared with itself, and is excluded and counted instead."""
    xs, ys, same, total = [], [], 0, 0
    for a, sa, b, sb in zip(scores1, served1, scores2, served2):
        if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
            continue
        total += 1
        if set(sa or {}) & set(sb or {}) or not sa or not sb:
            same += 1
            continue
        xs.append(a)
        ys.append(b)
    out = {"n": len(xs), "n_pairs_scored": total, "n_same_model": same,
           "mean_abs_delta": None, "pearson": None}
    if not xs:
        out["note"] = (f"not computed: same model served both judges "
                       f"({same}/{total} scored pairs)" if total else
                       "not computed: no record was scored by both judges")
        return out
    out["mean_abs_delta"] = round(sum(abs(a - b) for a, b in zip(xs, ys)) / len(xs), 3)
    r = pearson(xs, ys)
    out["pearson"] = round(r, 3) if r is not None else None
    if same:
        out["note"] = f"{same}/{total} scored pairs excluded: same model served both judges"
    return out
