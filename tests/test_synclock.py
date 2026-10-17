"""Inter-process sync lock: a second concurrent sync skips instead of double-ingesting."""
import os
import threading
import time

import pytest

from regintel import config, ingest, synclock


@pytest.fixture(autouse=True)
def _state_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STATE", tmp_path / "state")


def test_lock_is_exclusive_and_released():
    with synclock.sync_lock():
        assert synclock.lock_path().exists()
        with pytest.raises(synclock.SyncBusy, match="already running"):
            with synclock.sync_lock():
                pass
    assert not synclock.lock_path().exists()
    with synclock.sync_lock():          # free again after release
        pass


def test_lock_released_when_the_sync_raises():
    with pytest.raises(ValueError):
        with synclock.sync_lock():
            raise ValueError("boom")
    assert not synclock.lock_path().exists()


def test_stale_lock_from_a_crashed_run_is_broken():
    synclock.lock_path().parent.mkdir(parents=True, exist_ok=True)
    synclock.lock_path().write_text('{"pid": 1, "token": "dead"}', encoding="utf-8")
    old = time.time() - synclock.STALE_S - 60
    os.utime(synclock.lock_path(), (old, old))
    with synclock.sync_lock():          # no SyncBusy
        assert "dead" not in synclock.lock_path().read_text(encoding="utf-8")
    assert not synclock.lock_path().exists()


def test_fresh_foreign_lock_is_respected_and_not_removed():
    synclock.lock_path().parent.mkdir(parents=True, exist_ok=True)
    synclock.lock_path().write_text('{"pid": 1, "token": "other"}', encoding="utf-8")
    with pytest.raises(synclock.SyncBusy):
        with synclock.sync_lock():
            pass
    assert synclock.lock_path().exists()  # a lock we do not own is never deleted


def test_second_concurrent_sync_skips_with_message(monkeypatch):
    started, release = threading.Event(), threading.Event()
    ran = []

    def slow_sync(source, limit, rebuild, refresh, progress):
        ran.append(source)
        started.set()
        release.wait(5)
        return {"rss": {"new_docs": 1}, "backfill": {"new_docs": 0}, "failures": 0}

    monkeypatch.setattr(ingest, "_sync", slow_sync)
    first: dict = {}
    t = threading.Thread(target=lambda: first.update(ingest.sync(progress=lambda *_: None)))
    t.start()
    assert started.wait(5)
    msgs: list[str] = []
    second = ingest.sync(progress=msgs.append)          # returns at once, does not wait
    release.set()
    t.join(5)

    assert second["skipped"] is True and "already running" in second["reason"]
    assert any("Sync skipped" in m for m in msgs)
    assert second["rss"]["new_docs"] == 0 and second["failures"] == 0   # callers' keys intact
    assert first["rss"]["new_docs"] == 1 and not first.get("skipped")
    assert ran == ["rss"]                                # the pass ran exactly once
    assert not synclock.lock_path().exists()


def test_progress_refreshes_the_heartbeat():
    with synclock.sync_lock() as lock:
        old = time.time() - 600
        os.utime(lock.path, (old, old))
        lock.wrap(lambda *_: None)("tick")
        assert time.time() - lock.path.stat().st_mtime < 60
