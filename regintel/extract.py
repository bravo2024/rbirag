"""Layout-aware PDF text extraction (PyMuPDF).

RBI documents are government PDFs: multi-level numbering, tables with rulings,
footnotes in smaller type, running letterheads/headers/footers, occasional
two-column front matter and dot-leader tables of contents. Plain text dumps
interleave those and break retrieval, so this extractor:

  * reads lines in reading order with their coordinates (sort=True);
  * detects ruled tables (drawings gate + find_tables) and emits rows as
    "cell | cell" lines;
  * treats lines in the top/bottom band as margins and drops those that repeat
    across pages (letterheads, page numbers, footers);
  * separates smaller-type footnote lines and appends them as Notes per page;
  * skips table-of-contents pages (most lines carry dot leaders);
  * rejoins words hyphenated across line breaks.

Output: [{"page": int, "text": str, "footnotes": [...]}]
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from pathlib import Path

HEADER_BAND = 0.09   # fraction of page height searched as running header
FOOTER_BAND = 0.07
REPEAT_SHARE = 0.25  # margin line repeated on this share of pages -> dropped
MIN_RULING_OBJECTS = 8
FOOTNOTE_SIZE_RATIO = 0.1  # smaller than body by >10% -> footnote candidate

ALLOWED_NON_ASCII = set("\u20b9\u2013\u2014\u2018\u2019\u201c\u201d\u2022\u2026\u00b0\u00b1\u00d7\u2265\u2264\u00a7\u00bd\u00bc\u00be\u00b7\u2192\u2212")
LETTERHEAD_RE = re.compile(
    r"^(RESERVE BANK OF INDIA|www\.rbi\.org\.in|Reserve Bank of India)$"
    r"|^(Department of|Central Office|Tel No|Tel:|Fax|e-mail|Email:|Telephone|Maker Tower|World Trade)"
    r"|continuation sheet\s*$|^Withdrawn$", re.I)
PAGE_NUMBER_RE = re.compile(
    r"^(page\s+)?\d{1,3}(\s*(of|/)\s*\d{1,3})?$|^[-\u2013]\s*\d{1,3}\s*[-\u2013]$|^[ivxlc]{1,6}$", re.I)
LEADER_RE = re.compile(r"(\.\s?){5,}|_{6,}")


class ExtractionError(RuntimeError):
    pass


def is_garbled(line: str) -> bool:
    """Mis-encoded Hindi letterhead lines (RBI PDFs carry a Hindi masthead)."""
    weird = sum(1 for c in line if ord(c) > 127 and c not in ALLOWED_NON_ASCII)
    return weird >= 3 and weird / max(len(line), 1) > 0.05


def norm_key(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\d+", "#", text.lower())).strip()


def _clean_line(line: str) -> str:
    line = unicodedata.normalize("NFKC", line)
    line = line.replace("\u00ad", "")
    return re.sub(r"[ \t\u00a0]+", " ", line).strip()


def _table_rows(table) -> list[str]:
    rows = []
    for row in table.extract():
        cells = [re.sub(r"\s+", " ", (c or "")).strip() for c in row]
        cells = [c for c in cells if c]
        if cells:
            rows.append(" | ".join(cells))
    return rows


def _body_font_size(page_dict: dict) -> float:
    sizes: Counter[float] = Counter()
    for block in page_dict["blocks"]:
        if block.get("type") != 0:
            continue
        for ln in block["lines"]:
            for sp in ln["spans"]:
                sizes[round(sp["size"], 1)] += len(sp["text"].strip())
    return sizes.most_common(1)[0][0] if sizes else 0.0


def _overlap(a, b) -> float:
    inter = a & b
    if inter.is_empty:
        return 0.0
    return (inter.width * inter.height) / max(a.width * a.height, 1e-6)


def _page_items(page) -> list[tuple[str, str, float, float, bool]]:
    """(kind, text, y0, y1, in_margin) in reading order; kind: text|table|footnote."""
    import fitz

    height = page.rect.height
    tables = []
    try:
        if len(page.get_drawings()) >= MIN_RULING_OBJECTS:
            tables = [t for t in page.find_tables().tables
                      if t.row_count >= 2 and t.col_count >= 2]
    except Exception:  # find_tables can fail on odd pages
        tables = []
    boxes = [fitz.Rect(t.bbox) for t in tables]
    emitted: set[int] = set()
    items: list[tuple[str, str, float, float, bool]] = []
    pdict = page.get_text("dict", sort=True)
    body = _body_font_size(pdict)
    for block in pdict["blocks"]:
        if block.get("type") != 0:
            continue
        for ln in block["lines"]:
            text = "".join(sp["text"] for sp in ln["spans"])
            if not text.strip():
                continue
            x0, y0, x1, y1 = ln["bbox"]
            lrect = fitz.Rect((x0, y0, x1, y1))
            hit = next((i for i, b in enumerate(boxes) if _overlap(lrect, b) > 0.6), None)
            if hit is not None:
                if hit not in emitted:
                    emitted.add(hit)
                    for row in _table_rows(tables[hit]):
                        items.append(("table", row, y0, y0, False))
                continue
            margin = y1 < height * HEADER_BAND or y0 > height * (1 - FOOTER_BAND)
            small = body and all(sp["size"] <= body * (1 - FOOTNOTE_SIZE_RATIO)
                                 for sp in ln["spans"] if sp["text"].strip())
            kind = "footnote" if (small and y0 > height * 0.55 and not margin) else "text"
            items.append((kind, text, y0, y1, margin))
    return items


def _dehyphenate(text: str) -> str:
    return re.sub(r"(?<=[a-z])-\n(?=[a-z])", "", text)


def extract_pdf(path: str | Path) -> list[dict]:
    import fitz

    try:
        doc = fitz.open(path)
    except Exception as exc:  # noqa: BLE001
        raise ExtractionError(f"cannot open {path}: {exc}") from exc

    raw_pages = [_page_items(p) for p in doc]
    n_pages = len(raw_pages)

    # Margin lines that repeat across pages are running headers/footers.
    margin_counts: Counter[str] = Counter()
    for items in raw_pages:
        for seen in {norm_key(t) for kind, t, _, _, margin in items if margin and kind == "text"}:
            if seen:
                margin_counts[seen] += 1
    repeated = {k for k, c in margin_counts.items()
                if c >= max(3, REPEAT_SHARE * n_pages)}

    pages: list[dict] = []
    for idx, items in enumerate(raw_pages, start=1):
        out: list[str] = []
        notes: list[str] = []
        prev_y1, prev_h = None, 10.0
        for kind, raw, y0, y1, margin in items:
            line = _clean_line(raw)
            if not line or is_garbled(line):
                continue
            if kind == "footnote":
                if notes and not re.match(r"^(\d{1,2}|\*|\u2020|\u2021)[).]?\s", line):
                    notes[-1] += " " + line
                else:
                    notes.append(line)
                continue
            if margin and (norm_key(line) in repeated or PAGE_NUMBER_RE.match(line)):
                continue
            if LETTERHEAD_RE.search(line):
                continue
            if prev_y1 is not None and y0 - prev_y1 > 0.55 * prev_h and out and out[-1]:
                out.append("")
            out.append(line)
            prev_y1, prev_h = y1, max(y1 - y0, 6.0)
        # bare page numbers just outside the margin band
        nz = [i for i, ln in enumerate(out) if ln]
        for pos in (nz[0] if nz else None, nz[-1] if nz else None):
            if pos is not None and re.fullmatch(r"\d{1,3}", out[pos]):
                out[pos] = ""
        # TOC pages: most lines carry dot leaders
        nonblank = [ln for ln in out if ln]
        leaders = sum(1 for ln in nonblank if LEADER_RE.search(ln))
        if nonblank and leaders / len(nonblank) > 0.35:
            continue
        out = [ln for ln in out if not LEADER_RE.search(ln)]
        text = _dehyphenate("\n".join(out).strip())
        if text:
            page = {"page": idx, "text": text}
            if notes:
                page["footnotes"] = notes
            pages.append(page)
    doc.close()
    if not pages:
        raise ExtractionError(f"no extractable text in {path} (scanned image PDF?)")
    return pages


def full_text(pages: list[dict], with_footnotes: bool = True) -> str:
    parts = []
    for p in pages:
        parts.append(p["text"])
        if with_footnotes and p.get("footnotes"):
            parts.append("Notes: " + " ".join(p["footnotes"]))
    return "\n\n".join(parts)
