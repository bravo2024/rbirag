"""Inter-process lock around ``ingest.sync``.

Two app sessions with auto-sync (or the app plus the CI job) would otherwise
both ingest the same new documents and append them twice to the JSONL files.
Cross-platform, no dependency: an ``O_CREAT | O_EXCL`` lockfile under
data/state. A second sync does not wait — it skips with a clear message. A
lock whose last heartbeat is older than ``STALE_S`` (a crashed run) is broken;
a live sync refreshes the heartbeat on every progress message.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

from regintel import config

STALE_S = 30 * 60


class SyncBusy(RuntimeError):
    """Another sync holds the lock."""


def lock_path() -> Path:
    return config.STATE / "sync.lock"


def _describe(path: Path) -> str:
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
        age = int(time.time() - path.stat().st_mtime)
        return f"pid {info.get('pid')} on {info.get('host')} started {info.get('started')} (active {age}s ago)"
    except Exception:  # noqa: BLE001 - unreadable/just-removed lock
        return "another process"


class _Held:
    def __init__(self, path: Path, token: str):
        self.path, self.token = path, token

    def heartbeat(self) -> None:
        try:
            os.utime(self.path, None)
        except OSError:
            pass

    def wrap(self, progress):
        """Progress callback that also refreshes the lock heartbeat."""
        def _p(*a, **k):
            self.heartbeat()
            return progress(*a, **k)
        return _p


def _try_create(path: Path, token: str) -> bool:
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump({"pid": os.getpid(), "host": os.environ.get("COMPUTERNAME")
                   or os.environ.get("HOSTNAME", ""), "token": token,
                   "started": time.strftime("%Y-%m-%dT%H:%M:%S")}, fh)
    return True


@contextmanager
def sync_lock(stale_s: float | None = None):
    """Hold the sync lock or raise SyncBusy (never blocks)."""
    stale_s = STALE_S if stale_s is None else stale_s
    path = lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    token = f"{os.getpid()}-{time.time_ns()}"
    if not _try_create(path, token):
        try:
            age = time.time() - path.stat().st_mtime
        except FileNotFoundError:
            age = 0.0
        if age > stale_s:  # crashed run: break it once, then race fairly
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        if age <= stale_s or not _try_create(path, token):
            raise SyncBusy(f"a sync is already running ({_describe(path)}); skipping this one")
    held = _Held(path, token)
    try:
        yield held
    finally:
        try:  # only remove a lock that is still ours
            if json.loads(path.read_text(encoding="utf-8")).get("token") == token:
                path.unlink()
        except Exception:  # noqa: BLE001
            pass
