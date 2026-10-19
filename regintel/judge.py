"""LLM-judge harness for answer-level (RAGAS) evaluation.

The answer eval scores generated answers with RAGAS metrics judged by a
separate LLM. This module is the single integration point:

*   **Provider plumbing.** `RegIntelJudgeLLM` is a ragas ``BaseRagasLLM`` that
    routes every judge call through ``regintel.llm.chat``. Judges therefore
    inherit the whole production stack for free: provider failover, pinned
    judge selection (``pin=("gemini", "gemini-3.8-flash")``), per-model
    rate-limit holds, and the sqlite response cache — re-running an eval is
    free and instant.

*   **ragas 0.4.3 compatibility shim.** ``ragas.llms.base`` hard-imports two
    langchain-community shims (``chat_models.vertexai.ChatVertexAI`` and
    ``llms.VertexAI``) that langchain-community >= 0.4 removed in its
    sunset cleanup. Both symbols are used only inside isinstance checks, so
    stubbing them is behaviour-preserving. ``install_ragas_compat()`` is
    idempotent and runs before the first ragas import.

*   **Metric choice.** Pinned ragas 0.4.3 keeps the classic metric classes
    (``ragas.metrics``), which accept any ``BaseRagasLLM`` and retry
    unparseable JSON through their fix-output prompt. The v2
    ``ragas.metrics.collections`` metrics require an instructor LLM that
    calls the provider directly, bypassing the failover chain and cache
    above — revisit at ragas v1.

The app never imports this module; only eval scripts and tests do.
"""
from __future__ import annotations

import math
import os
import re
import sys
import time
import types
import warnings
from collections import Counter
from dataclasses import dataclass, field

# RAGAS telemetry can block on a blackholed endpoint. Disable it before any
# ragas import; this affects evaluation only, never the app's model calls.
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

from langchain_core.outputs import Generation, LLMResult

from regintel import llm as llm_py


# ---------------------------------------------------------------------------
# ragas 0.4.3 compatibility shim
# ---------------------------------------------------------------------------
def install_ragas_compat() -> dict[str, str]:
    """Stub the two legacy langchain-community symbols ragas 0.4.3 imports.

    Idempotent. Returns what was stubbed ({"symbol": "stub"}) for logging.
    """
    stubbed: dict[str, str] = {}

    try:
        from langchain_community.chat_models.vertexai import ChatVertexAI  # noqa: F401
    except Exception:  # noqa: BLE001
        import langchain_community.chat_models as _ccm

        mod = types.ModuleType("langchain_community.chat_models.vertexai")

        class ChatVertexAI:  # noqa: N801 - mirrors the real class name
            """isinstance-only stand-in; this project never calls Vertex."""

        mod.ChatVertexAI = ChatVertexAI  # type: ignore[attr-defined]
        sys.modules["langchain_community.chat_models.vertexai"] = mod
        _ccm.vertexai = mod  # type: ignore[attr-defined]
        stubbed["langchain_community.chat_models.vertexai.ChatVertexAI"] = "stub"

    try:
        from langchain_community.llms import VertexAI  # noqa: F401
    except Exception:  # noqa: BLE001
        import langchain_community.llms as _cl

        class VertexAI:  # noqa: N801
            """isinstance-only stand-in; this project never calls Vertex."""

        _cl.VertexAI = VertexAI  # type: ignore[attr-defined]
        stubbed["langchain_community.llms.VertexAI"] = "stub"

    return stubbed


install_ragas_compat()

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import ragas  # noqa: F401  (import after the shim)
    from ragas import RunConfig, evaluate
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms.base import BaseRagasLLM
    from ragas.metrics import (
        AnswerRelevancy,
        Faithfulness,
        LLMContextPrecisionWithoutReference,
    )

#: Metrics this harness runs, by output column name.
METRIC_CLASSES: dict[str, type] = {
    "faithfulness": Faithfulness,
    "answer_relevancy": AnswerRelevancy,
    "llm_context_precision_without_reference": LLMContextPrecisionWithoutReference,
}


# ---------------------------------------------------------------------------
# Judge LLM over the production provider stack
# ---------------------------------------------------------------------------
def _allow_exact_model(provider: str, model: str) -> None:
    """RAGAS-only: let an explicitly named judge model through discovery rank.

    The app's MODEL_RANK is intentionally narrow (only the newest/strongest
    models are auto-selected). A RAGAS judge chain is an explicit allowlist, so
    add the exact model id as a rank pattern for this process; the app's own
    model defaults and provider config are untouched.
    """
    patterns = llm_py.MODEL_RANK.get(provider)
    if patterns is None:
        return
    exact = re.escape(model)
    if exact not in patterns:
        patterns.insert(0, exact)


@dataclass
class RegIntelJudgeLLM(BaseRagasLLM):
    """A ragas LLM whose every call goes through ``regintel.llm.chat``.

    ``pin`` selects one judge (provider, model); failures fall over along the
    normal chain and rate limits are handled by the per-model holds in
    ``regintel.llm``.  ``judge_chain`` is a RAGAS-only alternative: an ordered
    list of exact ``(provider, model)`` pairs tried in turn, with no Groq or
    app-default fallback.  ``n > 1`` (e.g. AnswerRelevancy strictness) is served
    as n sequential single-completion calls so each one is cached on its own
    prompt.
    """

    pin: tuple[str, str] | None = None
    max_tokens: int = 6000
    temperature: float = 0.0
    strict_pin: bool = False
    #: Ordered exact models for RAGAS-only judging; takes precedence over pin.
    judge_chain: list[tuple[str, str]] | None = None
    #: (provider, model, cached) of every successful call.
    served: list = field(default_factory=list)
    #: Short per-model cooldown so a rate-limited first choice is not hammered.
    _cooldowns: dict[tuple[str, str], float] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        for provider, model in self.judge_chain or []:
            _allow_exact_model(provider, model)

    def _one(self, prompt: str) -> str:
        if self.judge_chain:
            return self._one_chain(prompt)
        kw = {"strict_pin": True} if self.strict_pin else {}
        call = llm_py.chat(
            [{"role": "user", "content": prompt}],
            temperature=self.temperature, max_tokens=self.max_tokens, pin=self.pin, **kw,
        )
        self.served.append((call["provider"], call["model"], call["cached"]))
        return call["text"]

    def _one_chain(self, prompt: str) -> str:
        """Try the RAGAS judge chain in order, one exact model at a time."""
        now = time.time()
        chain = [pair for pair in self.judge_chain
                 if self._cooldowns.get(pair, 0.0) <= now]
        if not chain:  # every judge briefly cooled down: retry the full order
            chain = list(self.judge_chain)
        errors: list[str] = []
        for provider, model in chain:
            try:
                call = llm_py.chat(
                    [{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    providers=[provider],
                    pin=(provider, model),
                    strict_pin=True,
                )
                self.served.append((call["provider"], call["model"], call["cached"]))
                self._cooldowns.pop((provider, model), None)
                return call["text"]
            except Exception as exc:  # noqa: BLE001 - try the next judge
                text = str(exc)
                errors.append(f"{provider}/{model}: {text[:180]}")
                transient = any(marker in text.lower()
                                for marker in ("429", "rate", "timeout", "temporarily"))
                self._cooldowns[(provider, model)] = time.time() + (30.0 if transient else 300.0)
        raise RuntimeError("RAGAS judge chain exhausted: " + " | ".join(errors))

    def generate_text(self, prompt, n: int = 1, temperature=None, stop=None,
                      callbacks=None) -> LLMResult:  # type: ignore[override]
        gens = [Generation(text=self._one(prompt.to_string())) for _ in range(max(1, n))]
        # single prompt with n generations (ragas reads generations[0][i])
        return LLMResult(generations=[gens])

    async def agenerate_text(self, prompt, n: int = 1, temperature=None, stop=None,
                             callbacks=None) -> LLMResult:  # type: ignore[override]
        return self.generate_text(prompt, n=n, temperature=temperature, stop=stop)

    def is_finished(self, response: LLMResult) -> bool:
        return True


def judge_llm(pin: tuple[str, str] | None = None, max_tokens: int = 6000,
              strict_pin: bool = False,
              judge_chain: list[tuple[str, str]] | None = None) -> RegIntelJudgeLLM:
    return RegIntelJudgeLLM(pin=pin, max_tokens=max_tokens, strict_pin=strict_pin,
                            judge_chain=judge_chain)


def judge_embeddings() -> LangchainEmbeddingsWrapper:
    """The engine's own bge-small encoder (query prefix included) as ragas
    embeddings — no external embedding API, and AnswerRelevancy is scored in
    the same vector space retrieval uses."""
    from regintel.lc.embeddings import RegIntelEmbeddings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return LangchainEmbeddingsWrapper(RegIntelEmbeddings())


# ---------------------------------------------------------------------------
# Evaluation run
# ---------------------------------------------------------------------------
def build_metrics(names: list[str]) -> list:
    unknown = [n for n in names if n not in METRIC_CLASSES]
    if unknown:
        raise ValueError(f"unknown metrics: {unknown}; known: {sorted(METRIC_CLASSES)}")
    return [METRIC_CLASSES[n]() for n in names]


def _num(v):
    """JSON-safe number: NaN -> None."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else round(f, 4)


def run_eval(samples: list[dict], judge_pin: tuple[str, str] | None = None,
             metrics: list[str] | None = None, max_workers: int = 1,
             timeout: int = 300, show_progress: bool = False,
             strict_pin: bool = False,
             judge_chain: list[tuple[str, str]] | None = None) -> dict:
    """Score ``samples`` (SingleTurnSample kwargs: user_input, response,
    retrieved_contexts) with the selected metrics.

    Serial by default (max_workers=1) so free-tier rate limits survive; every
    judge call is cached in ``data/cache/llm.sqlite`` so reruns are free.
    Samples are scored one at a time so each row can carry ``judge_served``:
    the {"provider/model": calls} census of the judge calls behind its score
    (the pin is only a request; failover may serve another model).

    ``judge_chain`` is a RAGAS-only exact ordered list of
    ``(provider, model)`` pairs; when set it replaces the normal provider
    fallback and no app-default models are called.

    Returns {"rows": [{user_input, metric: score, judge_served, ...}],
    "scores": {metric: mean|None}, "served": {"provider/model": calls}}.
    """
    metrics = metrics or list(METRIC_CLASSES)
    judge = judge_llm(judge_pin, strict_pin=strict_pin, judge_chain=judge_chain)
    embeddings = judge_embeddings()
    built = build_metrics(metrics)
    rows: list[dict] = []
    total: Counter = Counter()
    for s in samples:
        judge.served.clear()
        result = evaluate(
            dataset=EvaluationDataset(samples=[SingleTurnSample(**s)]),
            metrics=built,
            llm=judge,
            embeddings=embeddings,
            run_config=RunConfig(max_workers=max_workers, timeout=timeout,
                                 max_retries=3, max_wait=120),
            raise_exceptions=False,
            show_progress=show_progress,
        )
        rec = result.to_pandas().to_dict(orient="records")[0]
        row = {k: (_num(v) if k not in ("user_input", "response", "retrieved_contexts")
                   else v) for k, v in rec.items()}
        census = Counter(f"{p}/{m}" for p, m, _c in judge.served)
        row["judge_served"] = dict(census)
        total.update(census)
        rows.append(row)
    scores: dict[str, float | None] = {}
    for m in metrics:
        vals = [r.get(m) for r in rows if isinstance(r.get(m), (int, float))]
        scores[m] = round(sum(vals) / len(vals), 4) if vals else None
    return {"rows": rows, "scores": scores, "served": dict(total)}
