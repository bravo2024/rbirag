#!/usr/bin/env python3
"""End-to-end answer eval: RAGAS metrics + mechanical pipeline metrics.

Stage 1 regenerates the pipeline's answers for the gold set
(evals/queries.jsonl) with the answer model pinned, saving raw artifacts to
evals/results/answers_<ts>.json. The same harness accepts the v2 set via
``--file`` / ``--category``; unanswerable rows are excluded by default because
they need the abstention harness, not RAGAS. Stage 2 scores those answers with RAGAS
(faithfulness, answer relevancy, context precision without reference) judged
by a pinned judge LLM through the project's provider chain
(regintel.judge). Stage 3 adds mechanical metrics from the same runs (gold
doc in context, citation validation, number grounding, abstention, latency).
Stage 4 repeats faithfulness with a second judge and reports agreement.

Judge calls run serially and every call is cached in data/cache/llm.sqlite,
so reruns are free; rate limits are absorbed by the provider failover chain.

    python scripts/eval_answers.py --limit 2
    python scripts/eval_answers.py --file evals/eval_set_v2.jsonl --category answerable --limit 5
    python scripts/eval_answers.py --answers evals/results/answers_<ts>.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:  # Windows consoles default to cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

from regintel import answer as answer_mod  # noqa: E402
from regintel import config, judge, store, util  # noqa: E402
from regintel.eval_metrics import first_hit_rank, judge_agreement  # noqa: E402
from regintel.index import Engine, QueryFilters  # noqa: E402

EVAL_FILE = ROOT / "evals" / "queries.jsonl"
RESULTS_DIR = ROOT / "evals" / "results"

DEFAULT_ANSWER_PIN = "groq:openai/gpt-oss-120b"
DEFAULT_JUDGE_PIN = "groq:openai/gpt-oss-120b"      # fastest + most JSON-reliable under
#   sustained load (see eval_judge_probe.py); same family as the answer model, so the
#   flagship metric is cross-checked by a second judge below.
DEFAULT_JUDGE2_PIN = "gemini:gemini-3.7-flash"      # cross-family check on faithfulness
DEFAULT_METRICS = "faithfulness,answer_relevancy,llm_context_precision_without_reference"


def _pin(s: str | None) -> tuple[str, str] | None:
    if not s:
        return None
    provider, _, model = s.partition(":")
    return (provider.strip(), model.strip())


def _med(vals: list[float]) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    return round(s[len(s) // 2], 3)


_CONTEXT_ENCODING = None


def _context_encoder():
    """Lazy tiktoken encoder used only for the RAGAS context budget."""
    global _CONTEXT_ENCODING
    if _CONTEXT_ENCODING is None:
        import tiktoken
        _CONTEXT_ENCODING = tiktoken.get_encoding("cl100k_base")
    return _CONTEXT_ENCODING


def truncate_contexts(contexts: list[str], max_tokens: int,
                      scope: str = "total") -> list[str]:
    """RAGAS-only context limit; never used by the app or retrieval.

    ``scope='total'`` fills the highest-ranked contexts first and truncates the
    last one to the remaining token budget. ``scope='per_context'`` caps each
    context independently. Token counts use cl100k_base as an approximation of
    the judge's tokenizer.
    """
    if max_tokens <= 0:
        return contexts
    enc = _context_encoder()
    if scope == "per_context":
        out = []
        for ctx in contexts:
            ids = enc.encode(ctx)
            out.append(ctx if len(ids) <= max_tokens else enc.decode(ids[:max_tokens]))
        return [c for c in out if c.strip()]

    remaining = max_tokens
    out = []
    for ctx in contexts:
        if remaining <= 0:
            break
        ids = enc.encode(ctx)
        if len(ids) <= remaining:
            out.append(ctx)
            remaining -= len(ids)
        else:
            out.append(enc.decode(ids[:remaining]))
            remaining = 0
            break
    return [c for c in out if c.strip()]


def apply_context_budget(samples: list[dict], max_tokens: int,
                         scope: str = "total") -> tuple[list[dict], dict]:
    """Return new RAGAS samples with a bounded context payload plus stats."""
    if max_tokens <= 0:
        return samples, {}
    enc = _context_encoder()
    before = sum(len(enc.encode(c)) for s in samples
                 for c in (s.get("retrieved_contexts") or []))
    out = []
    for s in samples:
        row = dict(s)
        row["retrieved_contexts"] = truncate_contexts(
            list(row.get("retrieved_contexts") or []), max_tokens, scope)
        out.append(row)
    after = sum(len(enc.encode(c)) for s in out
                for c in (s.get("retrieved_contexts") or []))
    return out, {
        "context_size": max_tokens,
        "context_scope": scope,
        "context_tokens_before": before,
        "context_tokens_after": after,
        "contexts_before": sum(len(s.get("retrieved_contexts") or []) for s in samples),
        "contexts_after": sum(len(s.get("retrieved_contexts") or []) for s in out),
    }


def stage_answers(cases: list[dict], answer_pin, strict_pin: bool = False) -> list[dict]:
    eng = Engine()
    records: list[dict] = []
    for i, case in enumerate(cases, 1):
        filt = None
        if case.get("institution") or case.get("as_of"):
            ents = (config.INSTITUTIONS[case["institution"]]["entities"]
                    if case.get("institution") else frozenset())
            filt = QueryFilters(entities=frozenset(ents), as_of=case.get("as_of"))
        t0 = time.perf_counter()
        res = eng.retrieve(case["query"], filters=filt, k=config.TOP_K)
        a = answer_mod.answer(case["query"], res["hits"],
                              low_confidence=bool(res["abstain"] and res["hits"]),
                              llm_pin=answer_pin, strict_pin=strict_pin)
        total_s = round(time.perf_counter() - t0, 2)
        docs = [h.doc_id for h in res["hits"]]
        gold = set(case["gold_doc_ids"])
        rank = first_hit_rank(docs, gold, len(docs))
        records.append({
            "query": case["query"], "note": case.get("note"),
            "institution": case.get("institution"), "gold_doc_ids": sorted(gold),
            "retrieved_doc_ids": docs, "gold_in_context": rank is not None, "gold_rank": rank,
            "contexts": [h.text for h in res["hits"]],
            "top_rerank": (res["hits"][0].scores.get("rerank") if res["hits"] else None),
            "retrieval_abstain": bool(res["abstain"]),
            "retrieval_timings_ms": res["trace"].get("timings_ms"),
            "mode": a["mode"], "abstain": a["abstain"], "answer_md": a["answer_md"],
            "claims": a["claims"], "claims_dropped": a["claims_dropped"],
            "claims_ungrounded": a["claims_ungrounded"], "parse_retries": a["parse_retries"],
            "answer_model": a["model"], "answer_provider": a["provider"],
            "answer_cached": a["cached"], "answer_latency_s": a["latency_s"],
            "answer_errors": a["errors"], "end_to_end_s": total_s,
        })
        print(f"[{i}/{len(cases)}] {a['mode']:9s} claims={len(a['claims'])} "
              f"dropped={a['claims_dropped']} ungrounded={a['claims_ungrounded']} "
              f"gold_in_ctx={records[-1]['gold_in_context']} retries={a['parse_retries']} "
              f"e2e={total_s}s  {case['query'][:52]}")
    return records


def stage_metric(samples: list[dict], judge_pin, metric: str, progress: bool = False,
                 strict_pin: bool = False, judge_chain=None, workers: int = 1):
    """Returns (scores, served): per-sample scores and, per sample, the
    {"provider/model": calls} census of the judge calls behind each score."""
    t0 = time.time()
    out = judge.run_eval(samples, judge_pin=judge_pin, metrics=[metric],
                         show_progress=progress, strict_pin=strict_pin,
                         max_workers=workers,
                         **({"judge_chain": judge_chain} if judge_chain else {}))
    vals = [r.get(metric) for r in out["rows"]]
    served = [r.get("judge_served") or {} for r in out["rows"]]
    nn = [v for v in vals if isinstance(v, (int, float))]
    mean = round(sum(nn) / len(nn), 3) if nn else None
    print(f"[judge] {metric}: mean={mean} over {len(nn)}/{len(vals)} scored  "
          f"({time.time() - t0:.0f}s)  served={out['served']}")
    return vals, served


def stage_agreement(samples: list[dict], f1: list, served1: list[dict], judge2_pin,
                    strict_pin: bool = False, workers: int = 1) -> dict:
    vals2, served2 = stage_metric(samples, judge2_pin, "faithfulness", strict_pin=strict_pin,
                                  workers=workers)
    out = judge_agreement(f1, served1, vals2, served2)
    out.update(judge2=":".join(judge2_pin or ()), scores_judge2=vals2, served_judge2=served2)
    return out


def census(per_record: list[dict]) -> dict[str, int]:
    """Sum {"provider/model": calls} dicts into one served-model census."""
    c: Counter = Counter()
    for d in per_record:
        c.update(d or {})
    return dict(c.most_common())


def mechanical_summary(records: list[dict]) -> dict:
    n = len(records)
    lat = [r["answer_latency_s"] for r in records]
    e2e = [r["end_to_end_s"] for r in records]
    ret_ms = [(r["retrieval_timings_ms"] or {}).get("total", 0) for r in records]
    claims = sum(len(r["claims"]) for r in records)
    dropped = sum(r["claims_dropped"] for r in records)
    ungrounded = sum(r["claims_ungrounded"] for r in records)
    return {
        "n_queries": n,
        "modes": {m: sum(1 for r in records if r["mode"] == m)
                  for m in ("llm", "abstain", "extractive")},
        "gold_doc_in_context_rate": round(sum(r["gold_in_context"] for r in records) / n, 3),
        "claims_total": claims,
        "claims_per_answer": round(claims / n, 2),
        "claims_dropped_by_validator": dropped,
        "claims_flagged_number_grounding": ungrounded,
        "claims_grounded_rate": round(1 - ungrounded / claims, 3) if claims else None,
        "parse_retries": sum(r["parse_retries"] for r in records),
        "retrieval_ms_median": _med(ret_ms),
        "answer_s_median": _med(lat),
        "end_to_end_s_median": _med(e2e),
    }


def write_markdown(path: Path, meta: dict, summary: dict, metric_means: dict,
                   per_query: list[dict], agreement: dict | None) -> None:
    lines = [f"# RBIRAG answer evaluation — {meta['ts']}", ""]
    judge_req = (" -> ".join(meta["judge_chain"]) if meta.get("judge_chain")
                 else meta["judge_pin"])
    ctx_req = (f"context {meta['context_size']} tok/{meta['context_scope']}"
               if meta.get("context_size") else "full context")
    lines += [f"Requested pins: answer `{meta['answer_pin']}` · judge `{judge_req}` · "
              f"judge 2 `{meta.get('judge2_pin')}` · strict pin: {meta.get('strict_pin', False)} · "
              f"{ctx_req} · "
              f"corpus `{meta.get('corpus_version')}` · k={meta['k']} · "
              f"scored {meta['n_scored']}/{meta['n_queries']} answers (mode=llm).", "",
              "**Served by** (pins are only requests; failover may serve other models):", "",
              f"- answers: {meta.get('answer_served')}",
              f"- judge (all metrics): {meta.get('judge_served')}",
              f"- judge 2 (faithfulness): {meta.get('judge2_served')}", ""]
    lines += ["## Answer-level metrics (RAGAS)", "",
              "| metric | mean | judged |", "|---|---|---|"]
    for m, v in metric_means.items():
        lines.append(f"| {m} | {v['mean']} | {v['n']}/{meta['n_scored']} |")
    lines += ["", "## Pipeline mechanics", "",
              f"- modes: {summary['modes']}",
              f"- gold doc in retrieved context: {summary['gold_doc_in_context_rate']:.1%}",
              f"- claims: {summary['claims_total']} total, "
              f"{summary['claims_per_answer']}/answer, validator dropped "
              f"{summary['claims_dropped_by_validator']}, number-grounding flagged "
              f"{summary['claims_flagged_number_grounding']} "
              f"(grounded rate {summary['claims_grounded_rate']})",
              f"- JSON parse retries (truncation guard): {summary['parse_retries']}",
              f"- latency medians: retrieval {summary['retrieval_ms_median']} ms · "
              f"answer {summary['answer_s_median']} s · end-to-end {summary['end_to_end_s_median']} s",
              ""]
    if agreement:
        lines += ["## Judge agreement (faithfulness)", ""]
        if agreement["n"]:
            lines.append(f"- second judge `{agreement['judge2']}`: n={agreement['n']} "
                         f"(different model served each judge), mean |Δ| = "
                         f"{agreement['mean_abs_delta']}, Pearson r = {agreement['pearson']}")
        if agreement.get("note"):
            lines.append(f"- {agreement['note']}")
        lines.append("")
    lines += ["## Per query", "",
              "| # | mode | faith. | relev. | ctx prec. | gold rank | claims | e2e s | query |",
              "|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(per_query, 1):
        q = r["query"].replace("|", "/")[:60]
        lines.append(f"| {i} | {r['mode']} | {r.get('faithfulness')} | {r.get('answer_relevancy')} "
                     f"| {r.get('llm_context_precision_without_reference')} | {r['gold_rank']} "
                     f"| {len(r['claims'])} | {r['end_to_end_s']} | {q} |")
    lines += [""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="RBIRAG answer eval (RAGAS + mechanical)")
    ap.add_argument("--file", default=str(EVAL_FILE),
                    help="JSONL questions (default evals/queries.jsonl; v2 schema accepted)")
    ap.add_argument("--category", action="append",
                    choices=["answerable", "amendment", "as_of", "unanswerable"],
                    help="v2 category filter (repeatable); default excludes unanswerable")
    ap.add_argument("--reviewed-only", action="store_true",
                    help="score only v2 rows whose review_status is 'reviewed'")
    ap.add_argument("--limit", type=int, default=0, help="first N queries after filtering (0 = all)")
    ap.add_argument("--answer-pin", default=DEFAULT_ANSWER_PIN)
    ap.add_argument("--judge-pin", default=DEFAULT_JUDGE_PIN)
    ap.add_argument("--judge-model", action="append", default=None,
                    help="RAGAS-only judge chain entry provider:model (repeatable); "
                         "when set, exactly these models are tried in order and Groq "
                         "is not a fallback")
    ap.add_argument("--judge2-pin", default=DEFAULT_JUDGE2_PIN)
    ap.add_argument("--skip-agreement", action="store_true")
    ap.add_argument("--strict-pin", action="store_true",
                    help="no fallback: answer and judge calls must be served by exactly "
                         "the pinned models (a call the pin cannot serve fails)")
    ap.add_argument("--metrics", default=DEFAULT_METRICS)
    ap.add_argument("--context-size", type=int, default=0,
                    help="RAGAS-only context budget in tokens (0 = full retrieved contexts)")
    ap.add_argument("--context-scope", choices=["total", "per_context"], default="total",
                    help="apply --context-size across all contexts or per context")
    ap.add_argument("--answers", default=None, help="reuse a saved answers_*.json")
    ap.add_argument("--tag", default="")
    ap.add_argument("--progress", action="store_true", help="ragas progress bar")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel judge calls (1 = serial, survives free-tier limits)")
    args = ap.parse_args(argv)

    answer_pin, judge_pin = _pin(args.answer_pin), _pin(args.judge_pin)
    judge_chain = [_pin(s) for s in (args.judge_model or [])]
    judge_chain = [p for p in judge_chain if p and p[0] and p[1]]
    if args.judge_model and len(judge_chain) != len(args.judge_model):
        ap.error("--judge-model entries must be provider:model")
    judge_label = (" -> ".join(f"{p}:{m}" for p, m in judge_chain)
                   if judge_chain else args.judge_pin)
    metric_names = [m.strip() for m in args.metrics.split(",") if m.strip()]
    ts = util.iso_now().replace(":", "-")
    suffix = f"_{args.tag}" if args.tag else ""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- stage 1: answers ---------------------------------------------------
    if args.answers:
        blob = json.loads(Path(args.answers).read_text(encoding="utf-8"))
        records = blob["records"]
        print(f"loaded {len(records)} answers from {args.answers}")
    else:
        cases = [json.loads(l) for l in Path(args.file).read_text(encoding="utf-8").splitlines()
                 if l.strip()]
        is_v2 = any("category" in c for c in cases)
        if args.category:
            cases = [c for c in cases if c.get("category") in args.category]
        elif is_v2:
            cases = [c for c in cases
                     if c.get("category") in ("answerable", "amendment", "as_of")]
            print("v2 set: scoring answerable + amendment + as_of; "
                  "unanswerable rows need abstention eval, not RAGAS")
        if args.reviewed_only:
            cases = [c for c in cases if c.get("review_status") == "reviewed"]
        if args.limit:
            cases = cases[:args.limit]
        if not cases:
            print("no cases selected")
            return 1
        print(f"== stage 1: {len(cases)} answers from {Path(args.file).name} "
              f"(pin {answer_pin}) ==")
        records = stage_answers(cases, answer_pin, args.strict_pin)
        ans_path = RESULTS_DIR / f"answers_{ts}{suffix}.json"
        ans_path.write_text(json.dumps({
            "ts": ts, "answer_pin": args.answer_pin, "k": config.TOP_K,
            "strict_pin": args.strict_pin,
            "corpus_version": (store.read_manifest() or {}).get("version"),
            "records": records}, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {ans_path.relative_to(ROOT)}")

    # ---- stage 2: RAGAS scores ---------------------------------------------
    samples = [{"user_input": r["query"], "response": r["answer_md"],
                "retrieved_contexts": r["contexts"]}
               for r in records if r["mode"] == "llm"]
    scored_records = [r for r in records if r["mode"] == "llm"]
    context_meta: dict = {}
    if args.context_size:
        samples, context_meta = apply_context_budget(
            samples, args.context_size, args.context_scope)
        print(f"RAGAS context budget: {args.context_size} tokens ({args.context_scope}); "
              f"{context_meta['context_tokens_before']} -> "
              f"{context_meta['context_tokens_after']} context tokens; "
              f"{context_meta['contexts_before']} -> {context_meta['contexts_after']} contexts")
    metric_means: dict[str, dict] = {}
    if samples:
        print(f"\n== stage 2: RAGAS on {len(samples)} answers (judge {judge_label}) ==")
        for m in metric_names:
            try:
                vals, served = stage_metric(samples, judge_pin, m, progress=args.progress,
                                            strict_pin=args.strict_pin,
                                            judge_chain=judge_chain,
                                            workers=args.workers)
            except Exception as exc:  # noqa: BLE001 - one metric must not kill the run
                print(f"[judge] {m} FAILED: {str(exc)[:200]}")
                vals, served = [None] * len(samples), [{}] * len(samples)
            nn = [v for v in vals if isinstance(v, (int, float))]
            metric_means[m] = {"mean": round(sum(nn) / len(nn), 3) if nn else None,
                               "n": len(nn)}
            for rec, v, sv in zip(scored_records, vals, served):
                rec[m] = v
                rec.setdefault("judge_served", {})[m] = sv
    else:
        print("no llm-mode answers to score")

    # ---- stage 3: agreement -------------------------------------------------
    agreement = None
    if samples and not args.skip_agreement and "faithfulness" in metric_means:
        print(f"\n== stage 3: agreement (judge2 {args.judge2_pin}) ==")
        try:
            agreement = stage_agreement(
                samples, [r.get("faithfulness") for r in scored_records],
                [r["judge_served"]["faithfulness"] for r in scored_records],
                _pin(args.judge2_pin), strict_pin=args.strict_pin)
        except Exception as exc:  # noqa: BLE001 - keep the run's other results
            print(f"[judge] agreement FAILED: {str(exc)[:200]}")

    # ---- stage 4: assemble --------------------------------------------------
    summary = mechanical_summary(records)
    meta = {"ts": ts, "answer_pin": args.answer_pin,
            "judge_pin": judge_label if judge_chain else args.judge_pin,
            "judge_chain": [f"{p}:{m}" for p, m in judge_chain] if judge_chain else None,
            "judge_chain_strict": bool(judge_chain),
            "k": config.TOP_K, "n_queries": len(records), "n_scored": len(samples),
            "corpus_version": (store.read_manifest() or {}).get("version"),
            "judge2_pin": args.judge2_pin, "strict_pin": args.strict_pin,
            "answer_served": dict(Counter(
                f"{r['answer_provider']}/{r['answer_model']}" for r in records
                if r["answer_model"]).most_common()),
            "judge_served": census([sv for r in scored_records
                                    for sv in r.get("judge_served", {}).values()]),
            "judge2_served": census((agreement or {}).get("served_judge2", [])),
            **context_meta}
    out_path = RESULTS_DIR / f"answer_eval_{ts}{suffix}.json"
    out_path.write_text(json.dumps({"meta": meta, "metric_means": metric_means,
                                    "mechanical": summary, "agreement": agreement,
                                    "records": records}, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    md_path = RESULTS_DIR / f"answer_eval_{ts}{suffix}.md"
    write_markdown(md_path, meta, summary, metric_means, records, agreement)

    print("\n== headline ==")
    for m, v in metric_means.items():
        print(f"  {m:45s} {v['mean']}  (n={v['n']})")
    print(f"  gold doc in context ................ {summary['gold_doc_in_context_rate']:.1%}")
    print(f"  modes .............................. {summary['modes']}")
    if agreement:
        if agreement["n"]:
            print(f"  judge agreement .................... n={agreement['n']} "
                  f"|Δ|={agreement['mean_abs_delta']} r={agreement['pearson']}")
        print(f"  {agreement.get('note', '')}")
    print(f"  answers served by .................. {meta['answer_served']}")
    print(f"  judge served by .................... {meta['judge_served']}")
    print(f"  judge 2 served by .................. {meta['judge2_served']}")
    print(f"\nwrote {out_path.relative_to(ROOT)} and {md_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
