"""Provider-agnostic LLM synthesis with multi-provider failover, live model
discovery and a disk cache.

Answer synthesis is OPTIONAL — retrieval with citations is the core product.
With no keys configured the keyed providers are skipped instantly, the
keyless anonymous gateway (Kilo) is tried, and with nothing reachable the app
drops to extractive mode. A single OpenAI-compatible endpoint can be forced
via OPENAI_BASE_URL / OPENAI_API_KEY / LLM_MODEL.

The failover chain and the live-model registry are the same methods proven in
the other projects in this portfolio (agenticresearch, TAXRAGENHANCE):

* Keyed providers lead (Groq -> Gemini free tier -> NVIDIA Build), then the
  keyless Kilo gateway; override with REGINTEL_LLM_ORDER.
* Model ids are discovered live from each provider's OpenAI-compatible
  /models endpoint (cached 10 minutes, ranked strongest-first, free-tier
  locked for Gemini) instead of trusting a hardcoded list — free models get
  renamed and retired constantly. The curated list is the fallback when
  discovery is unreachable.
* A model that fails is parked: 30 min for permanent errors (bad id, no
  access), 10 min after a timeout (it is queueing), 60 s after a rate limit —
  so one dead endpoint cannot slow the next answer.

All calls pass through a sqlite cache (data/cache/llm.sqlite) so evaluations
and repeated questions are reproducible and cheap.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from regintel import config

PROVIDERS: dict[str, dict] = {
    "groq": {
        "base": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "max_tokens": 32768,
        "models": ["openai/gpt-oss-120b"],
    },
    # Gemini API, free tier only. Free vs paid is decided by the Google Cloud
    # project behind the key: a key from an AI Studio project with no billing
    # account linked can never be charged (over quota it returns 429, which
    # fails over to the next provider). Only Flash / Flash-Lite text models
    # are ever selected; live discovery decides the newest.
    "gemini": {
        "base": "https://generativelanguage.googleapis.com/v1beta/openai",
        "key_env": "GEMINI_API_KEY",
        "max_tokens": 8192,
        "models": ["gemini-2.5-flash", "gemini-2.5-flash-lite"],
        "prefer_newest": True,
    },
    "nvidia-build": {
        "base": "https://integrate.api.nvidia.com/v1",
        "key_env": "NVIDIA_API_KEY",
        "max_tokens": 8192,
        # Nemotron-3 answers in ~1s on the free endpoints; DeepSeek queues
        # past 60s there, so it comes after (2026-10-07).
        "models": ["nvidia/nemotron-3-ultra-550b-a55b", "nvidia/nemotron-3-super-120b-a12b",
                   "deepseek-ai/deepseek-v4.1-flash"],
    },
    "kilo-free": {
        "base": "https://api.kilo.ai/api/openrouter",
        "key_env": "KILO_API_KEY",
        "anonymous": True,          # blank key works — public gateway
        "max_tokens": 1500,
        "models": ["nvidia/nemotron-3-ultra-550b-a55b:free", "stepfun/step-3.7-flash:free",
                   "nvidia/nemotron-3-super-120b-a12b:free",
                   "inclusionai/ling-3.0-flash-sante:free", "poolside/laguna-s-2.1:free",
                   "kilo-auto/free"],
        # Reasoning models: keep the hidden reasoning out of the answer and
        # give it room, otherwise it eats max_tokens and the reply is empty
        # or leaks thoughts.
        "reasoning_exclude": True,
        "app_attribution": True,
    },
    # Kept configurable but left out of the default order: this free tier now
    # rejects calls from outside its own client (HTTP 403, checked 2026-10-02).
    "opencode-free": {
        "base": "https://opencode.ai/zen/v1",
        "key_env": "OPENCODE_ZEN_API_KEY",
        "anonymous": True,
        "models": ["deepseek-v4-flash-free", "mimo-v2.5-free", "ling-3.0-flash-free"],
    },
}

# Auto mode walks this order; keyed providers lead (fast, strong, and skipped
# instantly without a key), then the keyless free gateway.
ORDER = [p.strip() for p in os.environ.get(
    "REGINTEL_LLM_ORDER", "groq,gemini,nvidia-build,kilo-free").split(",") if p.strip()]

CACHE_PATH = config.DATA / "cache" / "llm.sqlite"


# ---------------------------------------------------------------------------
# Live model registry (ported from agenticresearch)
# ---------------------------------------------------------------------------
_MODELS_CACHE: dict[str, tuple[float, list[str]]] = {}
_MODELS_TTL = 600  # seconds; a miss is not cached, so the next call retries

# Providers advertise far more than chat models, and the strongest ones are
# not listed first. Discovered ids are ranked by these patterns (earlier =
# tried first); anything matching MODEL_SKIP is not a chat model at all.
MODEL_RANK: dict[str, list[str]] = {
    "groq": [r"gpt-oss-120b", r"kimi-k", r"llama-3\.3-70b", r"qwen", r"gpt-oss-20b",
             r"llama-3\.1-8b"],
    "gemini": [r"^gemini-\d+(\.\d+)?-flash$", r"^gemini-\d+(\.\d+)?-flash-lite$"],
    "nvidia-build": [r"nemotron-3-ultra", r"nemotron-3-super", r"deepseek-v4", r"kimi-k3",
                     r"qwen3", r"llama-3\.3-70b"],
    "kilo-free": [r"nemotron-3-super", r"step-3", r"nemotron-3-ultra", r"ling-3", r"laguna-s",
                  r"kilo-auto/free", r"openrouter/free"],
}
MODEL_SKIP = re.compile(
    r"embed|guard|safety|safeguard|retriever|reward|vlm|vision|omni|ocr|whisper|tts|clip|"
    r"parse|orpheus|allam|audio|image|coder-6|codellama|llama2|lfm-2|nano", re.I)
MODELS_PER_PROVIDER = 5  # failover tries at most this many per provider

# (provider, model) -> epoch seconds until it may be tried again.
# (provider, "*") blocks the whole provider (bad key).
_MODEL_HOLD: dict[tuple[str, str], float] = {}
_PERMANENT_HTTP = {400, 401, 402, 403, 404, 410, 422}
_HOLD_PERMANENT_S = 1800
_HOLD_TRANSIENT_S = 60
_HOLD_SLOW_S = 600  # a model that timed out is queueing; give it 10 minutes
_HOLD_WAIT_MAX_S = 90  # full blackout: wait this long for the earliest hold to
                       # lift, then re-plan once before giving up
TIMEOUT = "timeout"
FREE_TIER_LOCKED = {"gemini"}  # only MODEL_RANK matches may ever be called


def _lookup_secret(name: str) -> str:
    """Read one key from Streamlit secrets.

    Accepts both layouts people actually use in the Cloud editor: a flat
    top-level ``GROQ_API_KEY = "..."`` and a single section such as
    ``[api_keys]`` holding the same key. Streamlit raises when no secrets
    file exists at all, so every access is guarded.
    """
    try:
        import streamlit as st  # noqa: PLC0415

        secrets = st.secrets
    except Exception:  # noqa: BLE001 - not running under Streamlit
        return ""
    try:
        value = secrets.get(name, "")
        if value:
            return str(value)
    except Exception:  # noqa: BLE001
        pass
    # One level of nesting: [api_keys] / [keys] / [secrets] GROQ_API_KEY = ...
    try:
        for section in list(secrets):
            try:
                block = secrets[section]
            except Exception:  # noqa: BLE001
                continue
            getter = getattr(block, "get", None)
            if getter is None:
                continue
            value = getter(name, "")
            if value:
                return str(value)
    except Exception:  # noqa: BLE001
        pass
    return ""


def _key_from(env_name: str) -> tuple[str, str]:
    """(key, source) with source in {"env", "secrets", "missing"}."""
    key = os.environ.get(env_name, "")
    if key:
        return key, "env"
    key = _lookup_secret(env_name)
    if key:
        return key, "secrets"
    return "", "missing"


def _get_key(env_name: str) -> str:
    return _key_from(env_name)[0]


def attribution_headers(cfg: dict) -> dict:
    """Optional client-attribution headers. OFF by default: sending Kilo's own
    Referer / X-Title makes this app pass as Kilo's client, which it is not.
    REGINTEL_KILO_ATTRIBUTION=1 opts in explicitly (e.g. if Kilo ever documents
    a required header for the anonymous gateway)."""
    if cfg.get("app_attribution") and os.environ.get("REGINTEL_KILO_ATTRIBUTION") == "1":
        return {"HTTP-Referer": "https://kilo.ai/", "X-Title": "Kilo Code"}
    return {}


def _provider_headers(cfg: dict, key: str) -> dict:
    headers = {"Authorization": f"Bearer {key or 'anonymous'}", "Content-Type": "application/json"}
    headers.update(attribution_headers(cfg))
    return headers


def available() -> bool:
    if os.environ.get("OPENAI_BASE_URL"):
        return True
    return any(cfg.get("anonymous") or _get_key(cfg["key_env"]) for cfg in PROVIDERS.values())


# ---------------------------------------------------------------------------
# Live model discovery
# ---------------------------------------------------------------------------
def discover_models(provider: str, timeout: int = 10, force: bool = False) -> list[str]:
    """Model ids from the provider's OpenAI-compatible /models endpoint.

    Cached for 10 minutes. Returns [] when the provider is unknown, the
    endpoint is down, or a keyed provider has no key. A miss is not cached,
    so the next call tries again. ``force`` refetches even inside the TTL
    (long-run refresher; see refresh_model_registry_if_stale).
    """
    cfg = PROVIDERS.get(provider)
    if not cfg:
        return []
    now = time.time()
    hit = _MODELS_CACHE.get(provider)
    if not force and hit and now - hit[0] < _MODELS_TTL:
        return hit[1]
    key = _get_key(cfg["key_env"])
    if not cfg.get("anonymous") and not key:
        return []
    ids: list[str] = []
    try:
        r = requests.get(f"{cfg['base'].rstrip('/')}/models",
                         headers=_provider_headers(cfg, key), timeout=timeout)
        r.raise_for_status()
        ids = [m.get("id") for m in r.json().get("data", []) if m.get("id")]
        # Gemini's OpenAI-compatible list says "models/gemini-2.5-flash";
        # chat/completions wants the bare id.
        ids = [i[len("models/"):] if i.startswith("models/") else i for i in ids]
    except Exception:  # noqa: BLE001
        ids = []
    if ids:
        _MODELS_CACHE[provider] = (now, ids)
    return ids


def _allowed(pname: str, model: str) -> bool:
    if pname not in FREE_TIER_LOCKED:
        return True
    return any(re.search(p, model, re.I) for p in MODEL_RANK.get(pname, []))


def _usable(pname: str, model: str) -> bool:
    now = time.time()
    return _MODEL_HOLD.get((pname, "*"), 0) < now and _MODEL_HOLD.get((pname, model), 0) < now


def _earliest_hold_left() -> float | None:
    """Seconds until the soonest failure hold lifts, or None when none are held.

    When every candidate model is held (a rate-limit storm parked the whole
    chain), callers get an *empty* call order and would otherwise fail without
    a single attempt. The remaining hold time lets ``chat`` wait that window
    out instead.
    """
    now = time.time()
    left = [t - now for t in _MODEL_HOLD.values() if t > now]
    return min(left) if left else None


def _rank_models(pname: str, ids: list[str]) -> list[str]:
    cfg = PROVIDERS.get(pname, {})
    ids = [i for i in ids if not MODEL_SKIP.search(i)]
    if cfg.get("anonymous"):
        ids = [i for i in ids if "free" in i.lower()]
    ranked, seen = [], set()
    for pat in MODEL_RANK.get(pname, []):
        rx = re.compile(pat, re.I)
        for i in ids:
            if i not in seen and rx.search(i):
                ranked.append(i)
                seen.add(i)
    if cfg.get("prefer_newest"):
        # Within each family keep newest versions first: gemini-3.8-flash
        # before gemini-2.5-flash before gemini-2.5-flash-lite.
        def version(m: str) -> float:
            v = re.search(r"(\d+(?:\.\d+)?)", m)
            return float(v.group(1)) if v else 0.0

        def family(m: str) -> int:
            return next((k for k, pat in enumerate(MODEL_RANK.get(pname, []))
                         if re.search(pat, m, re.I)), 99)

        ranked.sort(key=lambda m: (family(m), -version(m)))
    return ranked


def _interleave(ranked: list[str], pname: str) -> list[str]:
    """Round-robin across model families so one rate-limited family cannot
    fill the whole per-provider budget. Gemini currently answers on
    flash-lite while every flash returns 429; without this the first five
    slots are all flash and flash-lite is never called."""
    families: dict[int, list[str]] = {}
    for m in ranked:
        fam = next((k for k, pat in enumerate(MODEL_RANK.get(pname, []))
                    if re.search(pat, m, re.I)), 99)
        families.setdefault(fam, []).append(m)
    queues = [families[k] for k in sorted(families)]
    out: list[str] = []
    while any(queues):
        for q in queues:
            if q:
                out.append(q.pop(0))
    return out


def _ordered_models(pname: str, live: list[str]) -> list[str]:
    """Models to try for one provider, best first.

    Curated ids lead even when live discovery omits them, so a known-good
    model is never dropped because /models paginated or listed it late.
    Gemini is the exception: its curated ids are older versions, so the
    ranked live list decides, rotated across families.
    """
    cfg = PROVIDERS.get(pname, {})
    ranked = _rank_models(pname, live)
    if cfg.get("prefer_newest"):
        return _interleave(ranked, pname)
    if cfg.get("anonymous"):
        # Free gateways rotate their catalogue weekly; the live list is the
        # only trustworthy source, and the curated list is the offline fallback.
        return ranked or list(cfg.get("models", []))
    out = [m for m in cfg.get("models", []) if _allowed(pname, m)]
    out += [m for m in ranked if m not in out]
    return out


def available_models(provider: str, free_only: bool = True) -> list[str]:
    """Live models for the picker, strongest first. Falls back to the
    curated list when discovery fails."""
    cfg = PROVIDERS.get(provider, {})
    static = list(cfg.get("models", []))
    discovered = discover_models(provider)
    if not discovered:
        return static
    ordered = _ordered_models(provider, discovered)
    if free_only and cfg.get("anonymous"):
        ordered = [m for m in ordered if "free" in m.lower()] or ordered
    return ordered or static


def _models_for(pname: str) -> list[str]:
    """Models the failover walks for one provider, without a network call:
    the ranked live list when discovery is cached, else the curated list."""
    cfg = PROVIDERS[pname]
    hit = _MODELS_CACHE.get(pname)
    if hit and time.time() - hit[0] < _MODELS_TTL:
        return _ordered_models(pname, hit[1])
    return list(cfg["models"])


def warm_model_registry() -> None:
    """Fetch every provider's /models in parallel so the first answer already
    walks live, ranked models instead of the built-in fallback lists."""
    with ThreadPoolExecutor(max(1, len(ORDER))) as ex:
        list(ex.map(discover_models, ORDER))


_REFRESH_TS = 0.0  # last self-heal attempt; chat() refreshes at most every
_REFRESH_MIN_GAP_S = 60  # minute, comfortably inside _MODELS_TTL


def refresh_model_registry_if_stale(max_age: float = _MODELS_TTL * 0.8) -> None:
    """Best-effort live refresh so long runs keep walking current models.

    Discovery is cached for _MODELS_TTL, but an eval runs for hours: without
    a refresh the failover silently falls back to the curated lists once the
    cache ages out and can serve calls with much older models than the live
    list offers. Called from chat() at most once a minute; a failed fetch is
    not cached, so the next attempt retries."""
    global _REFRESH_TS
    now = time.time()
    if now - _REFRESH_TS < _REFRESH_MIN_GAP_S:
        return
    stale = [p for p in ORDER
             if (hit := _MODELS_CACHE.get(p)) is None or now - hit[0] >= max_age]
    if not stale:
        return
    _REFRESH_TS = now
    with ThreadPoolExecutor(max(1, len(stale))) as ex:
        list(ex.map(lambda p: discover_models(p, force=True), stale))


def provider_status() -> list[dict]:
    """One row per provider for the sidebar picker: key state and live model
    count. Discovery is cached for 10 minutes, and providers without a key
    are never called — so this stays cheap on re-render."""
    rows: list[dict] = []
    for pname in dict.fromkeys(ORDER + list(PROVIDERS)):
        cfg = PROVIDERS[pname]
        key, source = _key_from(cfg["key_env"])
        if cfg.get("anonymous"):
            key_state, source = "anonymous", "none"
        elif key:
            key_state = f"key set ({source})"
        else:
            key_state = "no key"
        live: list[str] = []
        if cfg.get("anonymous") or key:
            live = [m for m in available_models(pname) if _usable(pname, m)]
        rows.append({"provider": pname, "key": key_state, "key_source": source,
                     "live_models": len(live), "first": live[0] if live else ""})
    return rows


# ---------------------------------------------------------------------------
# Failure holds
# ---------------------------------------------------------------------------
def _status_of(exc: Exception):
    if isinstance(exc, requests.exceptions.Timeout):
        return TIMEOUT
    resp = getattr(exc, "response", None)
    if resp is not None:
        return resp.status_code
    m = re.search(r"HTTP (\d{3})", str(exc))
    if m:
        return int(m.group(1))
    if "timed out" in str(exc).lower() or "timeout" in str(exc).lower():
        return TIMEOUT
    return None


def _hold(pname: str, model: str, status) -> None:
    """Remember a failure so the next call skips this model for a while."""
    cfg = PROVIDERS.get(pname, {})
    if status in _PERMANENT_HTTP:
        hold = _HOLD_PERMANENT_S
    elif status == TIMEOUT:
        hold = _HOLD_SLOW_S
    else:
        hold = _HOLD_TRANSIENT_S
    _MODEL_HOLD[(pname, model)] = time.time() + hold
    if status == 401 and not cfg.get("anonymous"):
        _MODEL_HOLD[(pname, "*")] = time.time() + _HOLD_PERMANENT_S


def _call_order(providers: list[str] | None = None,
                pin: tuple[str, str] | None = None) -> list[tuple[str, dict, list[str]]]:
    """Providers to try, each with its usable models, capped per provider so
    one dead endpoint cannot stall an answer. A pinned provider is tried
    first, with its pinned model at the head of its list (even when live
    discovery does not list it). Recently failed models and parked providers
    are skipped."""
    names = list(providers or ORDER)
    if pin and pin[0] in PROVIDERS:
        names = [pin[0]] + [n for n in names if n != pin[0]]
    out: list[tuple[str, dict, list[str]]] = []
    for pname in names:
        cfg = PROVIDERS.get(pname)
        if not cfg or not _usable(pname, "*"):
            continue
        models = [m for m in _models_for(pname) if _usable(pname, m) and _allowed(pname, m)]
        if pin and pname == pin[0] and pin[1] and _allowed(pname, pin[1]) and _usable(pname, pin[1]):
            if pin[1] in models:
                models.remove(pin[1])
            models.insert(0, pin[1])
        out.append((pname, cfg, models[:MODELS_PER_PROVIDER]))
    return out


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
def _cache_conn() -> sqlite3.Connection:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(CACHE_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, response TEXT, model TEXT, ts REAL)")
    return conn


def _cache_key(model: str, messages: list[dict], max_tokens: int, temperature: float) -> str:
    payload = json.dumps({"m": model, "msgs": messages, "mt": max_tokens, "t": temperature},
                         sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Calls
# ---------------------------------------------------------------------------
def _post_chat(cfg: dict, key: str, model: str, messages: list[dict],
               max_tokens: int, temperature: float, timeout=(10, 45)) -> str:
    body: dict = {"model": model, "messages": messages,
                  "max_tokens": max_tokens, "temperature": temperature}
    if cfg.get("reasoning_exclude"):
        body["reasoning"] = {"effort": "low", "exclude": True}
    r = requests.post(f"{cfg['base'].rstrip('/')}/chat/completions",
                      headers=_provider_headers(cfg, key), json=body, timeout=timeout)
    if r.status_code in (429, 500, 502, 503, 504):
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:160]}")
    r.raise_for_status()
    data = r.json()
    if "error" in data and not data.get("choices"):
        raise RuntimeError(str(data["error"])[:200])
    msg = data["choices"][0]["message"]
    # Hidden-reasoning channels (reasoning_content / reasoning) are never
    # surfaced as the answer and never written to the cache: if the provider
    # returned only reasoning, the completion failed and the walk moves on to
    # the next model.
    text = (msg.get("content") or "").strip()
    if not text:
        raise RuntimeError("empty completion (no content; reasoning is never used)")
    return text


def chat(messages: list[dict], temperature: float = 0.2, max_tokens: int = 1400,
         providers: list[str] | None = None,
         pin: tuple[str, str] | None = None, strict_pin: bool = False) -> dict:
    """Try providers in order; first success wins. A pinned (provider, model)
    is tried first and the normal chain stays as the fallback. Returns
    {text, model, provider, cached, attempts}.

    A pin also scopes the cache: only the pinned model's own entry may answer
    from cache. Accepting any model's entry would let a call pinned to model B
    return the reply a fallback to model A cached for the same messages, so
    two "different" judges would silently be the same model.
    ``strict_pin`` (evals) additionally forbids live fallback: if the pinned
    model cannot serve the call, it raises instead of using another model."""
    attempts: list[str] = []
    if strict_pin:
        if not (pin and pin[0] and pin[1]):
            raise ValueError("strict_pin needs a full (provider, model) pin")
        # failure holds protect a *fallback chain*; with no fallback they would
        # only turn one slow call into a hold-long outage of every later call
        _MODEL_HOLD.pop((pin[0], pin[1]), None)

    # Long runs (hours of batch judging) outlive the discovery cache; refresh
    # before planning so the failover keeps preferring live, ranked models
    # over the curated fallback lists.
    refresh_model_registry_if_stale()

    def _plan() -> list[tuple[str, dict, list[str]]]:
        plan = _call_order(providers, pin)
        if strict_pin:  # the pinned model only: no other model, no custom endpoint
            return [(p, c, [pin[1]]) for p, c, ms in plan if p == pin[0] and pin[1] in ms]
        if os.environ.get("OPENAI_BASE_URL"):  # custom endpoint takes priority
            model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
            plan.insert(0, ("custom", {"base": os.environ["OPENAI_BASE_URL"],
                                       "key_env": "OPENAI_API_KEY"}, [model]))
        return plan

    wanted = _plan()

    # 0. a *full* blackout (every candidate model held) is a temporary rate-limit
    # storm: wait for the earliest hold to lift, then re-plan once. Bounded by
    # _HOLD_WAIT_MAX_S so a long hold (timeout/bad key) still fails fast.
    if not any(models for _, _, models in wanted):
        left = _earliest_hold_left()
        if left is not None and left <= _HOLD_WAIT_MAX_S:
            time.sleep(left + 0.5)
            wanted = _plan()

    # 1. cache lookup (same prompt => same answer). Unpinned: across every model
    # in the order. Pinned: the pinned model's entry only (even while it is
    # parked — a cache read costs nothing), or, for a provider-only pin, that
    # provider's models.
    if pin and pin[1]:
        lookup = [(pin[0], [pin[1]])]
    elif pin:
        lookup = [(p, ms) for p, _c, ms in wanted if p == pin[0]]
    else:
        lookup = [(p, ms) for p, _c, ms in wanted]
    conn = _cache_conn()
    for pname, models in lookup:
        for model in models:
            key = _cache_key(model, messages, max_tokens, temperature)
            row = conn.execute("SELECT response FROM cache WHERE key=?", (key,)).fetchone()
            if row:
                conn.close()
                return {"text": row[0], "model": model, "provider": pname, "cached": True,
                        "attempts": attempts}

    # 2. live calls, in order, until one succeeds
    last_exc: Exception | None = None
    for pname, cfg, models in wanted:
        key = _get_key(cfg["key_env"])
        if not cfg.get("anonymous") and not key:
            continue
        mt = min(max_tokens, int(cfg.get("max_tokens", max_tokens)))
        for model in models:
            for try_no in range(2):
                try:
                    text = _post_chat(cfg, key, model, messages, mt, temperature)
                    ck = _cache_key(model, messages, max_tokens, temperature)
                    conn.execute("INSERT OR REPLACE INTO cache (key, response, model, ts) VALUES (?,?,?,?)",
                                 (ck, text, model, time.time()))
                    conn.commit()
                    conn.close()
                    return {"text": text, "model": model, "provider": pname, "cached": False,
                            "attempts": attempts}
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    status = _status_of(exc)
                    attempts.append(f"{pname}/{model}: {status or type(exc).__name__}: {str(exc)[:110]}")
                    transient = status in (429, 500, 502, 503, 504)
                    if status == 429 and "exceeded your current quota" in str(exc).lower():
                        # Daily free-tier quota, not a per-minute limit: a 1.5 s
                        # retry cannot succeed and only doubles the wait.
                        transient = False
                    if transient and try_no == 0:
                        time.sleep(1.5)
                        continue
                    _hold(pname, model, status)
                    break
    conn.close()
    if strict_pin:
        why = last_exc or ("no key for the provider, model not allowed, or provider parked "
                           "after an authentication failure")
        raise RuntimeError(f"strict pin {pin[0]}:{pin[1]} could not serve the call "
                           f"(no fallback allowed): {why}; attempts={attempts}")
    raise RuntimeError(f"all providers failed: {last_exc}; attempts={attempts}")
