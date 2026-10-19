"""LangChain chat models over the same provider chain as regintel.llm.

Each provider in `regintel.llm.PROVIDERS` becomes one or more ChatOpenAI
instances (every provider speaks the OpenAI wire protocol); `.with_fallbacks()`
walks the same order the native layer uses, so LCEL chains inherit the
portfolio's proven failover without reimplementing it. LangChain's SQLite
cache keeps repeated questions and evals reproducible.
"""
from __future__ import annotations

from typing import Optional

from langchain_openai import ChatOpenAI

from regintel import config, llm


def _headers(cfg: dict) -> Optional[dict]:
    return llm.attribution_headers(cfg) or None


def _make(cfg: dict, key: str, model_id: str,
          temperature: float, max_tokens: int) -> ChatOpenAI:
    return ChatOpenAI(
        model=model_id, base_url=cfg["base"], api_key=key or "anonymous",
        temperature=temperature,
        max_tokens=min(max_tokens, int(cfg.get("max_tokens", max_tokens))),
        timeout=90, max_retries=1, default_headers=_headers(cfg),
        extra_body=({"reasoning": {"effort": "low", "exclude": True}}
                    if cfg.get("reasoning_exclude") else None),
    )


def build_chat_models(temperature: float = 0.2, max_tokens: int = 1400,
                      providers: Optional[list[str]] = None,
                      pin: Optional[tuple[str, str]] = None) -> list[ChatOpenAI]:
    """One ChatOpenAI per (provider, model) in failover order, keys applied.
    A pinned (provider, model) heads the chain, then the failover order walks
    two models per provider so a retired model id still fails over within the
    provider before moving on."""
    models: list[ChatOpenAI] = []
    if pin and pin[0] in llm.PROVIDERS:
        cfg = llm.PROVIDERS[pin[0]]
        key = llm._get_key(cfg["key_env"])
        if cfg.get("anonymous") or key:
            models.append(_make(cfg, key, pin[1], temperature, max_tokens))
    for name in (providers or llm.ORDER):
        cfg = llm.PROVIDERS.get(name)
        if not cfg:
            continue
        key = llm._get_key(cfg["key_env"])
        if not cfg.get("anonymous") and not key:
            continue
        for model_id in llm._models_for(name)[:2]:
            if pin and name == pin[0] and model_id == pin[1]:
                continue  # already at the head of the chain
            models.append(_make(cfg, key, model_id, temperature, max_tokens))
    if not models:
        raise RuntimeError("no LLM provider available (no key and no anonymous gateway)")
    return models


def chat_model(temperature: float = 0.2, max_tokens: int = 1400,
               providers: Optional[list[str]] = None,
               pin: Optional[tuple[str, str]] = None):
    """The failover chain as a single Runnable."""
    models = build_chat_models(temperature=temperature, max_tokens=max_tokens,
                               providers=providers, pin=pin)
    return models[0].with_fallbacks(models[1:]) if len(models) > 1 else models[0]


def enable_cache(path=None) -> None:
    """Route LCEL calls through a sqlite cache (mirrors the native layer)."""
    from langchain_community.cache import SQLiteCache
    from langchain_core.globals import set_llm_cache

    set_llm_cache(SQLiteCache(database_path=str(
        path or (config.DATA / "cache" / "llm_lc.sqlite"))))
