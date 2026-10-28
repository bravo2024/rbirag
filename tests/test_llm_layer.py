"""Provider registry: discovery, ranking, parking, call order — no network."""
from __future__ import annotations

import sys
import time
import types

from regintel import llm


class _Resp:
    def __init__(self, ids):
        self._ids = ids
        self.status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return {"data": [{"id": i} for i in self._ids]}


def test_discover_models_strips_gemini_prefix(monkeypatch):
    monkeypatch.setattr(llm, "_get_key", lambda name: "test-key")
    monkeypatch.setattr(llm.requests, "get",
                        lambda *a, **k: _Resp(["models/gemini-2.5-flash", "gemini-2.5-flash-lite"]))
    llm._MODELS_CACHE.pop("gemini", None)
    ids = llm.discover_models("gemini")
    assert ids == ["gemini-2.5-flash", "gemini-2.5-flash-lite"]
    llm._MODELS_CACHE.pop("gemini", None)


def test_rank_prefers_newest_flash():
    ids = ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-3.8-flash", "text-embedding-004"]
    ranked = llm._rank_models("gemini", ids)
    assert ranked == ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.5-flash-lite"]


def test_hold_parks_model():
    llm._MODEL_HOLD.clear()
    llm._hold("groq", "llama-3.3-70b-versatile", 429)
    assert not llm._usable("groq", "llama-3.3-70b-versatile")
    llm._MODEL_HOLD.clear()


def test_call_order_skips_parked_models():
    llm._MODEL_HOLD.clear()
    llm._MODELS_CACHE.pop("kilo-free", None)
    first = llm.PROVIDERS["kilo-free"]["models"][0]
    llm._hold("kilo-free", first, 400)
    order = llm._call_order(["kilo-free"])
    models = order[0][2]
    assert first not in models
    llm._MODEL_HOLD.clear()


def test_call_order_pins_provider_and_model_first():
    llm._MODEL_HOLD.clear()
    llm._MODELS_CACHE.pop("kilo-free", None)
    order = llm._call_order(pin=("kilo-free", "kilo-auto/free"))
    assert order[0][0] == "kilo-free"  # pinned provider leads the chain
    assert order[0][2][0] == "kilo-auto/free"  # pinned model heads its list
    llm._MODEL_HOLD.clear()


def test_call_order_pin_inserts_model_missing_from_discovery():
    llm._MODEL_HOLD.clear()
    llm._MODELS_CACHE["kilo-free"] = (time.time(), ["somebrand/new-model:free"])
    order = llm._call_order(pin=("kilo-free", "somebrand/new-model:free"))
    assert order[0][2][0] == "somebrand/new-model:free"
    llm._MODELS_CACHE.pop("kilo-free", None)
    llm._MODEL_HOLD.clear()


def test_chat_cache_hit_without_network(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setattr(llm, "CACHE_PATH", tmp_path / "llm.sqlite")
    monkeypatch.setattr(llm, "refresh_model_registry_if_stale", lambda *a, **k: None)
    monkeypatch.setattr(llm, "_call_order",
                        lambda providers=None, pin=None: [("kilo-free", llm.PROVIDERS["kilo-free"], ["m"])])

    def _boom(*a, **k):  # network must not be touched on a cache hit
        raise AssertionError("network call on cache hit")

    monkeypatch.setattr(llm.requests, "post", _boom)
    msgs = [{"role": "user", "content": "x"}]
    conn = llm._cache_conn()
    conn.execute("INSERT INTO cache (key, response, model, ts) VALUES (?,?,?,?)",
                 (llm._cache_key("m", msgs, 100, 0.2), "cached!", "m", 0.0))
    conn.commit()
    conn.close()
    out = llm.chat(msgs, max_tokens=100)
    assert out["cached"] is True and out["text"] == "cached!"


def test_cache_miss_marks_attempts(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setattr(llm, "CACHE_PATH", tmp_path / "llm.sqlite")
    monkeypatch.setattr(llm, "refresh_model_registry_if_stale", lambda *a, **k: None)
    monkeypatch.setattr(llm, "_get_key", lambda name: "test-key")
    monkeypatch.setattr(llm, "_call_order",
                        lambda providers=None, pin=None: [("kilo-free", llm.PROVIDERS["kilo-free"], ["m"])])
    monkeypatch.setattr(llm, "_post_chat",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("HTTP 429: busy")))
    llm._MODEL_HOLD.clear()
    try:
        llm.chat([{"role": "user", "content": "y"}], max_tokens=100)
        raise AssertionError("should have failed")
    except RuntimeError as exc:
        assert "attempts" in str(exc)
    llm._MODEL_HOLD.clear()


def test_chat_waits_out_full_blackout(monkeypatch, tmp_path):
    """Every model held at once (rate-limit storm) must not fail without any
    attempt: chat() waits for the earliest hold to lift, re-plans, and calls."""
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setattr(llm, "CACHE_PATH", tmp_path / "llm.sqlite")
    monkeypatch.setattr(llm, "refresh_model_registry_if_stale", lambda *a, **k: None)
    monkeypatch.setattr(llm, "_get_key", lambda name: "test-key")
    monkeypatch.setattr(llm, "_post_chat", lambda *a, **k: "recovered")
    llm._MODEL_HOLD.clear()
    t0 = time.time()
    try:
        for pname in dict.fromkeys(llm.ORDER + list(llm.PROVIDERS)):
            for model in llm._models_for(pname):
                llm._MODEL_HOLD[(pname, model)] = t0 + 0.15
        out = llm.chat([{"role": "user", "content": "hi"}], max_tokens=100)
        assert out["text"] == "recovered"
        assert time.time() - t0 >= 0.5  # it really waited the hold out
    finally:
        llm._MODEL_HOLD.clear()


def _chat_payload(message: dict) -> dict:
    class _Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": message}]}

    return _Resp()


def test_post_chat_rejects_reasoning_only_completions(monkeypatch):
    """Empty content + reasoning_content must fail the completion (the walk
    moves on) — hidden reasoning is never the answer and never cached."""
    monkeypatch.setattr(llm.requests, "post",
                        lambda *a, **k: _chat_payload({"content": "", "reasoning_content": "secret"}))
    monkeypatch.setattr(llm, "_provider_headers", lambda cfg, key: {})
    try:
        llm._post_chat({"base": "http://x"}, "k", "m", [{"role": "user", "content": "q"}], 10, 0.2)
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "reasoning is never used" in str(exc)


def test_post_chat_returns_content_even_with_reasoning_field(monkeypatch):
    monkeypatch.setattr(llm.requests, "post",
                        lambda *a, **k: _chat_payload({"content": "answer", "reasoning_content": "irrelevant"}))
    monkeypatch.setattr(llm, "_provider_headers", lambda cfg, key: {})
    assert llm._post_chat({"base": "http://x"}, "k", "m",
                          [{"role": "user", "content": "q"}], 10, 0.2) == "answer"


def test_refresh_model_registry_if_stale(monkeypatch):
    """Long runs must refetch expiring /models; otherwise the failover
    silently reverts to the curated fallback lists once the cache ages out."""
    monkeypatch.setattr(llm, "_REFRESH_TS", 0.0)
    monkeypatch.setattr(llm, "_MODELS_CACHE", {"groq": (time.time(), ["m1"])})
    calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(llm, "discover_models",
                        lambda provider, timeout=10, force=False: calls.append((provider, force)))
    llm.refresh_model_registry_if_stale(max_age=60)
    assert calls and all(force for _, force in calls)  # refetches stale providers
    assert ("groq", True) not in calls  # the fresh provider is left alone
    calls.clear()
    llm.refresh_model_registry_if_stale(max_age=60)  # throttled: once a minute
    assert calls == []


# ---------------------------------------------------------------------------
# Pinned calls: cache isolation + strict pin (no network, temp cache)
# ---------------------------------------------------------------------------
_MSGS = [{"role": "user", "content": "same prompt for both judges"}]


def _pin_setup(monkeypatch, tmp_path, models=("gemini-3.7-flash", "gemini-3.5-flash-lite")):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setattr(llm, "CACHE_PATH", tmp_path / "llm.sqlite")
    monkeypatch.setattr(llm, "refresh_model_registry_if_stale", lambda *a, **k: None)
    monkeypatch.setattr(llm, "_get_key", lambda name: "test-key")
    monkeypatch.setattr(llm, "_call_order", lambda providers=None, pin=None: [
        ("gemini", llm.PROVIDERS["gemini"], list(models))])
    llm._MODEL_HOLD.clear()


def _seed(model, text, msgs=_MSGS, max_tokens=100, temperature=0.2):
    conn = llm._cache_conn()
    conn.execute("INSERT INTO cache (key, response, model, ts) VALUES (?,?,?,?)",
                 (llm._cache_key(model, msgs, max_tokens, temperature), text, model, 0.0))
    conn.commit()
    conn.close()


def test_pinned_call_ignores_other_models_cache_entry(monkeypatch, tmp_path):
    """The 0.999-agreement bug: judge 1 fell back to flash-lite and cached its
    reply; judge 2, pinned to 3.7-flash with the same messages, must NOT get it."""
    _pin_setup(monkeypatch, tmp_path)
    _seed("gemini-3.5-flash-lite", "judge-1 reply (served by flash-lite)")
    seen = []
    monkeypatch.setattr(llm, "_post_chat",
                        lambda cfg, key, model, *a, **k: seen.append(model) or f"live:{model}")
    out = llm.chat(_MSGS, max_tokens=100, pin=("gemini", "gemini-3.7-flash"))
    assert out["cached"] is False
    assert out["model"] == "gemini-3.7-flash" and out["text"] == "live:gemini-3.7-flash"
    assert seen == ["gemini-3.7-flash"]


def test_unpinned_call_still_reads_any_model_cache(monkeypatch, tmp_path):
    """Product behaviour: without a pin the cache is shared across the order."""
    _pin_setup(monkeypatch, tmp_path)
    _seed("gemini-3.5-flash-lite", "cached by fallback")
    monkeypatch.setattr(llm, "_post_chat",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
    out = llm.chat(_MSGS, max_tokens=100)
    assert out["cached"] is True and out["model"] == "gemini-3.5-flash-lite"


def test_pinned_call_hits_pinned_models_own_cache_even_when_parked(monkeypatch, tmp_path):
    _pin_setup(monkeypatch, tmp_path, models=("gemini-3.5-flash-lite",))  # pin not in plan
    _seed("gemini-3.7-flash", "pinned entry")
    monkeypatch.setattr(llm, "_post_chat",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
    out = llm.chat(_MSGS, max_tokens=100, pin=("gemini", "gemini-3.7-flash"))
    assert out["cached"] is True and out["text"] == "pinned entry"
    assert (out["provider"], out["model"]) == ("gemini", "gemini-3.7-flash")


def test_pinned_fallback_is_served_but_not_replayed_to_other_pins(monkeypatch, tmp_path):
    """A non-strict pin may fall back live; the fallback's reply is cached under
    the fallback model only, so a later call pinned to that pin goes live again."""
    _pin_setup(monkeypatch, tmp_path)

    def post(cfg, key, model, *a, **k):
        if model == "gemini-3.7-flash":
            raise RuntimeError("HTTP 400: nope")
        return f"live:{model}"

    monkeypatch.setattr(llm, "_post_chat", post)
    out = llm.chat(_MSGS, max_tokens=100, pin=("gemini", "gemini-3.7-flash"))
    assert out["model"] == "gemini-3.5-flash-lite" and out["cached"] is False
    llm._MODEL_HOLD.clear()
    out2 = llm.chat(_MSGS, max_tokens=100, pin=("gemini", "gemini-3.7-flash"))
    assert out2["cached"] is False        # pinned model's entry is absent: no replay
    llm._MODEL_HOLD.clear()


def test_strict_pin_never_falls_back(monkeypatch, tmp_path):
    _pin_setup(monkeypatch, tmp_path)
    called = []

    def post(cfg, key, model, *a, **k):
        called.append(model)
        raise RuntimeError("HTTP 400: pinned model unavailable")

    monkeypatch.setattr(llm, "_post_chat", post)
    try:
        llm.chat(_MSGS, max_tokens=100, pin=("gemini", "gemini-3.7-flash"), strict_pin=True)
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "strict pin gemini:gemini-3.7-flash" in str(exc)
    assert called == ["gemini-3.7-flash"]  # the fallback model was never tried
    llm._MODEL_HOLD.clear()


def test_strict_pin_serves_pinned_model_and_ignores_custom_endpoint(monkeypatch, tmp_path):
    _pin_setup(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://custom")
    monkeypatch.setattr(llm, "_post_chat",
                        lambda cfg, key, model, *a, **k: f"live:{model}")
    out = llm.chat(_MSGS, max_tokens=100, pin=("gemini", "gemini-3.7-flash"), strict_pin=True)
    assert (out["provider"], out["model"]) == ("gemini", "gemini-3.7-flash")


def test_strict_pin_requires_full_pin(monkeypatch, tmp_path):
    _pin_setup(monkeypatch, tmp_path)
    for bad in (None, ("gemini", "")):
        try:
            llm.chat(_MSGS, pin=bad, strict_pin=True)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# No client impersonation
# ---------------------------------------------------------------------------
def test_kilo_attribution_headers_are_opt_in(monkeypatch):
    cfg = llm.PROVIDERS["kilo-free"]
    monkeypatch.delenv("REGINTEL_KILO_ATTRIBUTION", raising=False)
    h = llm._provider_headers(cfg, "")
    assert "HTTP-Referer" not in h and "X-Title" not in h
    monkeypatch.setenv("REGINTEL_KILO_ATTRIBUTION", "1")
    assert llm._provider_headers(cfg, "")["X-Title"] == "Kilo Code"
    assert "X-Title" not in llm._provider_headers(llm.PROVIDERS["groq"], "k")


def test_strict_pin_ignores_a_stale_hold_on_the_pinned_model(monkeypatch, tmp_path):
    """With no fallback, a hold from one slow call must not fail every later call."""
    _pin_setup(monkeypatch, tmp_path)
    llm._MODEL_HOLD[("gemini", "gemini-3.7-flash")] = time.time() + 600
    monkeypatch.setattr(llm, "_call_order", lambda providers=None, pin=None: [
        ("gemini", llm.PROVIDERS["gemini"],
         [m for m in ("gemini-3.7-flash", "gemini-3.5-flash-lite") if llm._usable("gemini", m)])])
    monkeypatch.setattr(llm, "_post_chat", lambda cfg, key, model, *a, **k: f"live:{model}")
    out = llm.chat(_MSGS, max_tokens=100, pin=("gemini", "gemini-3.7-flash"), strict_pin=True)
    assert out["model"] == "gemini-3.7-flash"
    llm._MODEL_HOLD.clear()


def test_strict_pin_unservable_says_why(monkeypatch, tmp_path):
    _pin_setup(monkeypatch, tmp_path)
    monkeypatch.setattr(llm, "_get_key", lambda name: "")   # no key => never attempted
    try:
        llm.chat(_MSGS, max_tokens=100, pin=("gemini", "gemini-3.7-flash"), strict_pin=True)
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "no fallback allowed" in str(exc) and "no key" in str(exc)


# ---------------------------------------------------------------------------
# Secrets: flat keys and one level of nesting (the Cloud editor's section form)
# ---------------------------------------------------------------------------
def _fake_streamlit(secrets):
    mod = types.ModuleType("streamlit")
    mod.secrets = secrets
    return mod


def test_key_from_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "env-key")
    assert llm._key_from("GROQ_API_KEY") == ("env-key", "env")


def test_key_from_flat_secrets(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "streamlit", _fake_streamlit({"GROQ_API_KEY": "flat"}))
    assert llm._key_from("GROQ_API_KEY") == ("flat", "secrets")


def test_key_from_nested_secrets(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "streamlit", _fake_streamlit(
        {"api_keys": {"GROQ_API_KEY": "nested"}, "other": {"X": "y"}}))
    assert llm._key_from("GROQ_API_KEY") == ("nested", "secrets")


def test_key_missing(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "streamlit", _fake_streamlit({}))
    assert llm._key_from("GROQ_API_KEY") == ("", "missing")


def test_key_survives_broken_secrets(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    class _Broken:
        @property
        def secrets(self):
            raise RuntimeError("No secrets found")

    monkeypatch.setitem(sys.modules, "streamlit", _Broken())
    assert llm._get_key("GROQ_API_KEY") == ""


def test_provider_status_reports_key_source(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "streamlit", _fake_streamlit(
        {"api_keys": {"GROQ_API_KEY": "nested"}}))
    monkeypatch.setattr(llm, "discover_models", lambda p, *a, **k: [])
    rows = {r["provider"]: r for r in llm.provider_status()}
    assert rows["groq"]["key"] == "key set (secrets)"
    assert rows["groq"]["key_source"] == "secrets"
    assert rows["kilo-free"]["key_source"] == "none"


def test_provider_status_does_not_count_the_static_fallback(monkeypatch):
    """A failed /models call must read as unavailable, not as N models: the
    static list is an offline fallback and opencode-free's ids are dead."""
    monkeypatch.setattr(llm, "discover_models", lambda p, *a, **k: [])
    rows = {r["provider"]: r for r in llm.provider_status()}
    assert rows["opencode-free"]["live_models"] == 0
    assert rows["opencode-free"]["first"] == ""


def test_provider_status_counts_live_models(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setattr(llm, "discover_models", lambda p, *a, **k: (
        ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"] if p == "groq" else []))
    rows = {r["provider"]: r for r in llm.provider_status()}
    assert rows["groq"]["live_models"] == 2
    assert rows["groq"]["first"] == "openai/gpt-oss-120b"


# ---------------------------------------------------------------------------
# Model ordering: curated first, and no single family can fill the budget
# ---------------------------------------------------------------------------
def test_curated_model_leads_even_when_discovery_omits_it(monkeypatch):
    llm._MODELS_CACHE["groq"] = (time.time(), ["qwen/qwen3.8-27b", "openai/gpt-oss-20b"])
    try:
        assert llm._models_for("groq")[0] == "openai/gpt-oss-120b"
    finally:
        llm._MODELS_CACHE.pop("groq", None)


def test_gemini_rotates_families_so_flash_lite_is_reachable(monkeypatch):
    ids = [f"gemini-{v}-flash" for v in ("3.8", "3.7", "3.6", "3.5", "2.5")] + \
          [f"gemini-{v}-flash-lite" for v in ("3.5", "3.1", "2.5")]
    llm._MODELS_CACHE["gemini"] = (time.time(), ids)
    try:
        models = llm._models_for("gemini")[:llm.MODELS_PER_PROVIDER]
    finally:
        llm._MODELS_CACHE.pop("gemini", None)
    assert "gemini-3.1-flash-lite" in models
    assert models[0] == "gemini-3.8-flash"
    assert len(set(models)) == len(models)