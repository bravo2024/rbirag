"""RSS feed parsing and diffing (offline, using a captured fixture)."""
import json

from conftest import FIXTURES
from regintel import rss


def _items():
    return rss.parse_feed((FIXTURES / "notifications_rss.xml").read_bytes())


def test_parse_feed_fields():
    items = _items()
    assert len(items) == 10
    first = items[0]
    assert first["pid"] and isinstance(first["pid"], int)
    assert len(first["title"]) > 10
    assert first["link"].startswith("https://www.rbi.org.in/")
    assert first["published"] and len(first["published"]) == 10
    assert len(first["summary"]) > 50


def test_new_items_diff():
    items = _items()
    pid = items[0]["pid"]
    seen = {"pages": {str(pid): {"title": "x"}}, "last_poll": None}
    new = rss.new_items(items, seen)
    assert pid not in [i["pid"] for i in new]
    assert len(new) == len(items) - 1
    # sorted oldest-first
    pub = [i["published"] for i in new]
    assert pub == sorted(pub)


def test_seen_roundtrip(tmp_path, monkeypatch):
    from regintel import config

    monkeypatch.setattr(config, "SEEN", tmp_path / "seen.json")
    seen = rss.load_seen()
    assert seen["pages"] == {}
    seen["pages"]["1"] = {"title": "t"}
    rss.save_seen(seen)
    again = json.loads((tmp_path / "seen.json").read_text(encoding="utf-8"))
    assert again["pages"]["1"]["title"] == "t"
    assert again["last_poll"] is not None
