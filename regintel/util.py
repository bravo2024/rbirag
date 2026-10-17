"""Small shared helpers: hashing, dates, titles, text normalisation."""
from __future__ import annotations

import datetime as dt
import hashlib
import re
from pathlib import Path

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], start=1)}
DATE_RE = re.compile(r"([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})")


def iso_now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_date(text: str | None) -> str | None:
    """First 'Month D, YYYY' in a string -> ISO date."""
    for m in DATE_RE.finditer(text or ""):
        mon = MONTHS.get(m.group(1).lower())
        if mon:
            try:
                return dt.date(int(m.group(3)), mon, int(m.group(2))).isoformat()
            except ValueError:
                continue
    return None


def clean_title(raw: str) -> str:
    """Normalise RBI titles: collapse whitespace, drop 'Updated as on/Effective
    from' clauses and unify dashes (RBI uses en/em dashes)."""
    t = re.sub(r"\s+", " ", raw or "").strip()
    t = re.sub(r"\(\s*Updated (?:as on|up to)[^)]*\)?", "", t, flags=re.I)
    t = re.sub(r"\(\s*Effective from[^)]*\)", "", t, flags=re.I)
    t = re.sub(r"\s+", " ", t).strip(" -")
    return t.replace("\u2013", "-").replace("\u2014", "-")


def norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def sentence_split(text: str) -> list[str]:
    """Split on sentence enders followed by a capital/quote/digit. Legal text
    friendly: keeps 'No. 27' style fragments together."""
    parts = re.split(r"(?<=[.;:])\s+(?=[A-Z(\[\u201c\u2018\"'])", text)
    return [p.strip() for p in parts if p.strip()]


def truncate(text: str, n: int) -> str:
    text = norm_ws(text)
    return text if len(text) <= n else text[: n - 1].rstrip() + "\u2026"
