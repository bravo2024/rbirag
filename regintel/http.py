"""Polite HTTP for rbi.org.in: browser User-Agent, throttling, retries, page cache.

RBI blocks default scripted clients; the browser UA plus a 1.2s per-host
throttle and retry/backoff keeps the crawler reliable and inoffensive.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import requests

from regintel import config

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
RETRIES = config.FETCH_RETRIES
MIN_INTERVAL = config.FETCH_THROTTLE_S
CACHE_DIR = config.CACHE_DIR

_session: requests.Session | None = None
_last_request = 0.0


class FetchError(RuntimeError):
    pass


def session() -> requests.Session:
    global _session
    if _session is None:
        s = requests.Session()
        s.headers.update({"User-Agent": USER_AGENT, "Accept": "*/*"})
        _session = s
    return _session


def _throttle() -> None:
    global _last_request
    wait = MIN_INTERVAL - (time.time() - _last_request)
    if wait > 0:
        time.sleep(wait)
    _last_request = time.time()


def get(url: str, headers: dict | None = None, retries: int | None = None,
        timeout: int = 30) -> requests.Response:
    """GET with retries on 429/5xx and connection errors."""
    last_exc: Exception | None = None
    attempts = retries if retries is not None else RETRIES
    for attempt in range(attempts):
        _throttle()
        try:
            r = session().get(url, headers=headers, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                last_exc = FetchError(f"HTTP {r.status_code} for {url}")
                time.sleep(min(2 ** attempt, 8))
                continue
            r.raise_for_status()
            return r
        except requests.RequestException as exc:  # includes timeouts
            last_exc = exc
            time.sleep(min(2 ** attempt, 8))
    raise FetchError(f"GET failed after {attempts} attempts: {url} ({last_exc})")


def _cache_path(url: str) -> Path:
    key = hashlib.sha1(url.encode("utf-8")).hexdigest()[:20]
    return CACHE_DIR / f"{key}.html"


def cached_text(url: str, max_age_hours: float | None = None) -> str:
    """Fetch a page with an on-disk cache (used for listing/view pages)."""
    path = _cache_path(url)
    if path.exists() and max_age_hours is not None:
        age_h = (time.time() - path.stat().st_mtime) / 3600
        if age_h <= max_age_hours:
            return path.read_text(encoding="utf-8", errors="replace")
    text = get(url).text
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def download(url: str, dest: Path, require_pdf: bool = True) -> dict:
    """Download a binary file; returns {"bytes", "sha256", "content_type"}."""
    body = get(url, headers={"Accept": "application/pdf,*/*"}).content
    if require_pdf and not body.startswith(b"%PDF"):
        raise FetchError(f"response is not a PDF (blocked or moved): {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(body)
    return {"bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest()}
