"""Poll RBI's notifications RSS feed.

The feed carries the ~10 most recent notifications (title, link, pubDate and a
description that restates the notification body head). We keep a local state of
every page id ever seen so each poll only surfaces new items.

Feed item -> https://www.rbi.org.in/scripts/NotificationUser.aspx?Id=<pid>&Mode=0
"""
from __future__ import annotations

import json
import re
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree as ET

from bs4 import BeautifulSoup

from regintel import config, http, util


def parse_feed(xml_bytes: bytes) -> list[dict]:
    """Parse the RSS XML into item dicts. Tolerates a BOM and empty descriptions."""
    root = ET.fromstring(xml_bytes.lstrip(b"\xef\xbb\xbf"))
    items: list[dict] = []
    for it in root.findall("./channel/item"):
        link = (it.findtext("link") or "").strip()
        m = re.search(r"[?&][Ii]d=(\d+)", link)
        pid = int(m.group(1)) if m else None
        html = it.findtext("description") or ""
        summary = re.sub(r"\s+", " ", BeautifulSoup(html, "html.parser").get_text(" ")).strip()
        try:
            published = parsedate_to_datetime(it.findtext("pubDate") or "").date().isoformat()
        except (TypeError, ValueError):
            published = None
        items.append({
            "pid": pid,
            "title": re.sub(r"\s+", " ", (it.findtext("title") or "").strip()),
            "link": link,
            "published": published,
            "summary": summary,
        })
    return items


def fetch_items() -> list[dict]:
    """Live poll of the RBI feed (one HTTP request)."""
    return parse_feed(http.get(config.RSS_URL).content)


def load_seen() -> dict:
    if config.SEEN.exists():
        return json.loads(config.SEEN.read_text(encoding="utf-8"))
    return {"pages": {}, "last_poll": None}


def save_seen(seen: dict) -> None:
    config.SEEN.parent.mkdir(parents=True, exist_ok=True)
    seen["last_poll"] = util.iso_now()
    config.SEEN.write_text(json.dumps(seen, indent=2, sort_keys=True), encoding="utf-8")


def new_items(items: list[dict], seen: dict) -> list[dict]:
    """Items whose page id has never been processed, newest last."""
    pages = seen.get("pages", {})
    out = [i for i in items if i["pid"] and str(i["pid"]) not in pages]
    out.sort(key=lambda i: (i["published"] or "", i["pid"] or 0))
    return out
