"""Backfill: crawl RBI's entity-wise Master Directions listings.

All 19 categories from RBI's Master Directions index (see config.LISTING_URLS):
8 cross-cutting pages (financial market, FEMA, payments, consumer protection,
…), 11 Department of Regulation entity pages (commercial banks … credit
information companies) and 11 Department of Supervision pages (the 2026
supervisory directions). Entries are de-duplicated by page id across listings.

Each entry links to a view page (BS_ViewMasDirections.aspx?id=<pid>) that exposes
the consolidated PDF on rbidocs.rbi.org.in. Backfill is a bootstrap: RSS keeps
the corpus current afterwards, and re-running backfill picks up new Directions.
"""
from __future__ import annotations

import re

from bs4 import BeautifulSoup

from regintel import config, http, util

TITLE_MIN_LEN = 12


def crawl_listing(name: str, url: str) -> list[dict]:
    """Anchors on a Master Directions listing page -> candidate entries."""
    html = http.cached_text(url, max_age_hours=6)
    soup = BeautifulSoup(html, "html.parser")
    found: dict[int, dict] = {}
    for a in soup.find_all("a", href=True):
        if "viewmasdirections" not in a["href"].lower():
            continue
        m = re.search(r"id=(\d+)", a["href"], re.I)
        title = util.norm_ws(a.get_text(" "))
        if not m or len(title) < TITLE_MIN_LEN or title.lower().startswith("http"):
            continue
        pid = int(m.group(1))
        if pid not in found or len(title) > len(found[pid]["listing_title"]):
            found[pid] = {"pid": pid, "listing_title": title,
                          "index": name, "index_url": url}
    return sorted(found.values(), key=lambda c: c["pid"])


def crawl_all() -> list[dict]:
    out: list[dict] = []
    for name, url in config.LISTING_URLS.items():
        out.extend(crawl_listing(name, url))
    return out


def resolve_view(pid: int) -> dict:
    """View page -> consolidated PDF link (first rbidocs notification PDF)."""
    url = config.VIEW_URL.format(pid=pid)
    html = http.cached_text(url, max_age_hours=6)
    soup = BeautifulSoup(html, "html.parser")
    pdfs = [a["href"] for a in soup.find_all("a", href=True)
            if re.search(r"rbidocs\.rbi\.org\.in/rdocs/notification/pdfs/.+\.pdf", a["href"], re.I)]
    if not pdfs:
        pdfs = [a["href"] for a in soup.find_all("a", href=True)
                if re.search(r"rbidocs.*\.pdf$", a["href"], re.I)
                and not re.search(r"Utkarsh|Accessib", a["href"], re.I)]
    return {"page_url": url,
            "pdf_url": pdfs[0].replace("http://", "https://") if pdfs else None}


def listing_meta(listing_title: str) -> dict:
    """'Updated as on' / 'Effective from' hints embedded in listing titles."""
    m = re.search(r"Updated (?:as on|up to)\s+([A-Za-z]+ \d{1,2},?,? \d{4})", listing_title, re.I)
    eff = re.search(r"Effective from\s+([A-Za-z]+ \d{1,2},? \d{4})", listing_title, re.I)
    return {"last_updated": util.parse_date(m.group(1)) if m else None,
            "effective_from": util.parse_date(eff.group(1)) if eff else None}
