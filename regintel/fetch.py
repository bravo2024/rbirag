"""Fetch stage: notification/view page -> official PDF + reference number.

The notification page (NotificationUser.aspx) embeds the canonical RBI
reference (e.g. RBI/2026-27/293) near its masthead and links the notification
PDF on rbidocs.rbi.org.in, which is the authoritative text we ingest.
"""
from __future__ import annotations

import re
from pathlib import Path

from bs4 import BeautifulSoup

from regintel import config, http

NOTIF_PDF_RE = re.compile(r"rbidocs.*notification/pdfs/.+\.pdf", re.I)
ANY_PDF_RE = re.compile(r"rbidocs.*\.pdf$", re.I)
REF_RE = re.compile(r"\bRBI/\d{4}-\d{2,4}/\d+\b")


def _pdf_links(soup: BeautifulSoup) -> list[str]:
    links = [a["href"] for a in soup.find_all("a", href=True) if NOTIF_PDF_RE.search(a["href"])]
    if not links:
        links = [a["href"] for a in soup.find_all("a", href=True)
                 if ANY_PDF_RE.search(a["href"]) and not re.search(r"Utkarsh|Accessib", a["href"], re.I)]
    return [l.replace("http://", "https://") for l in links]


def find_pdf(page_url: str, html: str | None = None) -> tuple[str | None, str | None]:
    """Returns (pdf_url, reference_number). `html` allows offline tests."""
    soup = BeautifulSoup(html if html is not None else http.get(page_url).text, "html.parser")
    links = _pdf_links(soup)
    text = re.sub(r"\s+", " ", soup.get_text(" "))
    ref = None
    k = text.find("Home Notifications")
    if k >= 0:
        m = REF_RE.search(text[k:k + 800])
        ref = m.group(0) if m else None
    if not ref:
        m = REF_RE.search(text)
        ref = m.group(0) if m else None
    return (links[0] if links else None), ref


def pdf_head_ref(pdf_path: Path, max_pages: int = 2) -> str | None:
    """Reference number from the PDF head as a fallback when the page lacks it."""
    try:
        import fitz  # PyMuPDF

        with fitz.open(pdf_path) as doc:
            text = "\n".join(doc[i].get_text() for i in range(min(max_pages, len(doc))))
        m = REF_RE.search(text)
        return m.group(0) if m else None
    except Exception:
        return None


def download_pdf(url: str, doc_id: str) -> dict:
    """Download to data/raw/<doc_id>.pdf; returns byte size + sha256."""
    dest = config.RAW / f"{doc_id}.pdf"
    return http.download(url, dest, require_pdf=True)
