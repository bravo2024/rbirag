#!/usr/bin/env python3
"""Judge selection probe: latency + JSON compliance per candidate LLM judge.

Every candidate is asked a RAGAS-shaped NLI question ("verdict per
statement") a few times with a nonce so the llm cache cannot mask latency.
Records wall time, whether the reply parsed as schema-valid JSON, which
(provider, model) actually answered (failover may step in), and any
attempt/error trail. Results land in evals/results/judge_probe_<ts>.json.

    python scripts/eval_judge_probe.py [--calls 3]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:  # Windows consoles default to cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

from regintel import llm, util  # noqa: E402

RESULTS_DIR = ROOT / "evals" / "results"

# Candidates to probe, in the order they would be tried as judge. The pinned
# model is attempted first; normal failover stays as fallback either way.
CANDIDATES: list[tuple[str, str | None]] = [
    ("groq", "openai/gpt-oss-120b"),
    ("gemini", None),                     # newest live flash, resolved below
    ("nvidia-build", "nvidia/nemotron-3-ultra-550b-a55b"),
    ("kilo-free", "nvidia/nemotron-3-ultra-550b-a55b:free"),
]

CONTEXT = (
    "KYC records must be updated at least once every two years for low-risk "
    "customers, once every eight years for medium-risk customers and once "
    "every two years for high-risk customers of regulated entities."
)
STATEMENTS = [
    "KYC records for low-risk customers must be refreshed at least once every two years.",
    "High-risk customer KYC must be refreshed at the same frequency as low-risk.",
    "The regulator permits a ten-year refresh cycle for medium-risk customers.",
]

_INSTRUCTION = (
    "Your task is to judge the faithfulness of statements based on a given context. "
    "For each statement return verdict 1 if the statement can be directly inferred from "
    "the context, else 0."
)
_SCHEMA = (
    '{"properties": {"statements": {"items": {"properties": {"statement": {"type": '
    '"string"}, "reason": {"type": "string"}, "verdict": {"type": "integer"}}, "required": '
    '["statement", "reason", "verdict"], "type": "object"}, "type": "array"}}, "required": '
    '["statements"], "type": "object"}'
)


def build_prompt(nonce: int) -> str:
    return (
        _INSTRUCTION
        + "\n\nPlease return the output in a JSON format that complies with the following "
          "schema as specified in JSON Schema:\n" + _SCHEMA
        + "\n\nContext:\n" + CONTEXT
        + "\n\nStatements:\n" + json.dumps(STATEMENTS)
        + "\n\nOutput:"
        + f"\n(nonce {nonce})"
    )


def parse_verdicts(text: str) -> dict | None:
    """Strict-ish parse of {"statements": [{"verdict": int, ...}]}."""
    start, depth = text.find("{"), 0
    if start < 0:
        return None
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    data = json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
                sts = data.get("statements")
                if isinstance(sts, list) and all(isinstance(s.get("verdict"), int)
                                                 and isinstance(s.get("statement"), str)
                                                 for s in sts):
                    return {"n_statements": len(sts), "verdicts": [s["verdict"] for s in sts]}
                return None
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="RegIntel judge selection probe")
    ap.add_argument("--calls", type=int, default=3, help="calls per candidate")
    args = ap.parse_args(argv)

    llm.warm_model_registry()  # live model lists for accurate fallbacks
    gemini_models = llm.available_models("gemini", free_only=True)

    rows: list[dict] = []
    for provider, model in CANDIDATES:
        pin_model = model or (gemini_models[0] if gemini_models else None)
        pin = (provider, pin_model) if pin_model else None
        print(f"\n== {provider}:{pin_model or '(auto)'} ==")
        for call_no in range(args.calls):
            prompt = build_prompt(time.time_ns())
            t0 = time.perf_counter()
            row = {"provider": provider, "pin_model": pin_model, "call": call_no + 1}
            try:
                res = llm.chat([{"role": "user", "content": prompt}],
                               temperature=0.0, max_tokens=700, pin=pin)
                ms = round((time.perf_counter() - t0) * 1000)
                parsed = parse_verdicts(res["text"])
                row.update(ms=ms, ok=True, json_valid=bool(parsed),
                           answered_by=f"{res['provider']}/{res['model']}",
                           cached=res["cached"], n_statements=(parsed or {}).get("n_statements"),
                           attempts=res["attempts"][-3:] if res["attempts"] else [])
                print(f"  call {call_no + 1}: {ms:6d} ms  json={'OK' if parsed else 'BAD'}  "
                      f"by {row['answered_by']}")
            except Exception as exc:  # noqa: BLE001
                ms = round((time.perf_counter() - t0) * 1000)
                row.update(ms=ms, ok=False, json_valid=False, error=str(exc)[:300])
                print(f"  call {call_no + 1}: {ms:6d} ms  FAILED  {str(exc)[:140]}")
            rows.append(row)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"judge_probe_{util.iso_now().replace(':', '-')}.json"
    out.write_text(json.dumps({"calls_per_candidate": args.calls, "rows": rows},
                              indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n== summary ==")
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["pin_model"] or r["provider"], []).append(r)
    for name, rs in by.items():
        ok = [r for r in rs if r["ok"]]
        jv = [r for r in rs if r.get("json_valid")]
        lat = sorted(r["ms"] for r in ok)
        med = lat[len(lat) // 2] if lat else None
        answered = sorted({r.get("answered_by", "?") for r in ok})
        print(f"{name:50s} ok {len(ok)}/{len(rs)}  json {len(jv)}/{len(rs)}  "
              f"median {med} ms  by {', '.join(answered)}")
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
