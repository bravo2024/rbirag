"""Structure-aware chunking of parsed RBI documents.

RBI texts are numbered: CHAPTERS, lettered headings, numbered paragraphs
("16."), decimal sub-paragraphs ("16.2"), nested clauses "(a)", "(i)", and
Explanation/Note blocks. The chunker builds that structure, then packs it into
chunks of roughly 120-380 words:

  * a paragraph is never cut mid-way when it has sub-clauses to split on;
  * small adjacent paragraphs under one chapter are merged;
  * oversized leaf paragraphs are split at sentence boundaries as a last resort.

Chunk records carry the paragraph label and heading path so citations can say
"Para 16.2, Chapter III, page 5" — the audit trail the UI renders.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from regintel import config, util

MIN_WORDS = config.CHUNK_MIN_WORDS
MAX_WORDS = config.CHUNK_MAX_WORDS
HARD_MAX_WORDS = config.CHUNK_HARD_MAX_WORDS

RE_CHAPTER = re.compile(
    r"^(CHAPTER|Chapter|SECTION|Section|PART|Part)\s*[-\u2013]?\s*([IVXLC]+|[A-Z]|\d{1,2})\b\s*[:\-\u2013\u2014.]?\s*(.*)$")
RE_ANNEX = re.compile(
    r"^(ANNEXURE|Annexure|ANNEX|Annex|APPENDIX|Appendix|SCHEDULE|Schedule)\s*[-\u2013:]?\s*"
    r"([IVXLC]+|\d{1,2}|[A-Z])?\b[\s:\-\u2013\u2014.]*(.*)$")
RE_TOP = re.compile(r"^(\d{1,3})\.\s+(\S.*)$")
RE_DEC = re.compile(r"^(\d{1,3}(?:\.\d{1,3})+)\.?\s+(\S.*)$")
RE_SUB = re.compile(r"^\(([a-z]{1,3}|\d{1,2}|[ivxlc]{1,4}|[A-Z]{1,2})\)\s*(.*)$")
RE_LETTER = re.compile(r"^([A-Z])\.\s+([A-Z][a-z].*)$")
RE_EXPL = re.compile(r"^(Explanation|Explanations|Note|Notes|Illustration|Illustrations|Example|Examples)"
                     r"\s*(\d{0,2})\s*([:.\-\u2013\u2014]|$)")
RE_ALONE = re.compile(r"^(\d{1,3}\.|\d{1,3}(?:\.\d{1,3})+\.?|\([a-zA-Z0-9]{1,4}\)|[A-Z]\.|[ivxlc]{1,4}\))$", re.I)
RE_PAGE_ONLY = re.compile(r"^\d{1,3}$")


def count_words(text: str) -> int:
    return len(text.split())


def render_lines(lines: list[tuple[str, int]]) -> str:
    """Join wrapped lines into paragraphs (blank-line separated); table rows and
    consecutive bullets stay tightly grouped with single newlines."""
    out: list[str] = []
    cur = ""
    last_kind: str | None = None

    def flush_para() -> None:
        nonlocal cur
        if cur:
            out.append(cur)
            cur = ""

    for text, _ in lines:
        if not text:
            flush_para()
            last_kind = None
            continue
        is_row = " | " in text or text.startswith(("\u2022", "-", "\u2013", "*"))
        if is_row:
            flush_para()
            if last_kind == "row" and out:
                out[-1] = out[-1] + "\n" + text
            else:
                out.append(text)
            last_kind = "row"
            continue
        cur = f"{cur} {text}" if cur else text
        last_kind = "para"
    flush_para()
    return "\n\n".join(out)


def prepare_lines(pages: list[dict]) -> list[tuple[str, int]]:
    """Flatten pages to (line, page); glue standalone numbers to the next line
    ("16." on one line and the paragraph text on the next)."""
    flat: list[tuple[str, int]] = []
    for pg in pages:
        for raw in pg["text"].split("\n"):
            flat.append((raw.strip(), int(pg.get("page", 0))))
    merged: list[tuple[str, int]] = []
    i = 0
    while i < len(flat):
        text, page = flat[i]
        if text and RE_ALONE.match(text):
            j = i + 1
            while j < len(flat) and not flat[j][0]:
                j += 1
            if j < len(flat) and flat[j][0] and not RE_ALONE.match(flat[j][0]):
                merged.append((f"{text} {flat[j][0]}", page))
                i = j + 1
                continue
        merged.append((text, page))
        i += 1
    return merged


@dataclass
class Unit:
    label: str = ""            # "16", "16.2", "(a)", "Explanation"
    section: str = ""          # chapter / annex heading path
    lines: list[tuple[str, int]] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)  # every unit label merged in

    def text(self) -> str:
        return render_lines(self.lines)

    def words(self) -> int:
        return count_words(self.text())

    def pages(self) -> tuple[int, int]:
        ps = [p for t, p in self.lines if t]
        return (min(ps), max(ps)) if ps else (0, 0)


def build_units(lines: list[tuple[str, int]]) -> list[Unit]:
    units: list[Unit] = []
    cur = Unit()
    section = ""

    def flush() -> None:
        nonlocal cur
        if cur.text().strip():
            if cur.label and not cur.labels:
                cur.labels = [cur.label]
            units.append(cur)
        cur = Unit(section=section)

    cur.section = section
    for text, page in lines:
        if not text:
            if cur.lines:
                cur.lines.append(("", page))
            continue
        if RE_PAGE_ONLY.match(text):
            continue
        m = RE_CHAPTER.match(text) or RE_ANNEX.match(text)
        if m:
            flush()
            section = util.norm_ws(text)[:120]
            cur = Unit(label="", section=section)
            cur.lines.append((text, page))
            continue
        m = RE_EXPL.match(text)
        if m and len(text) > len(m.group(0)) + 8:
            flush()
            cur.label = util.norm_ws(m.group(1) + (m.group(2) or ""))
            cur.lines.append((text, page))
            continue
        m = RE_TOP.match(text)
        if m:
            flush()
            cur.label = m.group(1)
            cur.lines.append((text, page))
            continue
        m = RE_DEC.match(text)
        if m:
            flush()
            cur.label = m.group(1)
            cur.lines.append((text, page))
            continue
        m = RE_SUB.match(text)
        if m and (m.group(2) or cur.words() >= 60):
            flush()
            cur.label = f"({m.group(1)})"
            cur.lines.append((text, page))
            continue
        m = RE_LETTER.match(text)
        if m and cur.words() >= 40:
            flush()
            cur.label = m.group(1)
            cur.lines.append((text, page))
            continue
        cur.lines.append((text, page))
    flush()
    return [u for u in units if u.text().strip()]


def _split_oversized(unit: Unit, limit: int) -> list[Unit]:
    """A leaf unit over the limit is split at sentence boundaries."""
    text = unit.text()
    paragraphs = text.split("\n")
    out: list[Unit] = []
    buf: list[str] = []
    buf_words = 0
    p_start, p_end = unit.pages()

    def spill() -> None:
        nonlocal buf, buf_words
        if buf:
            out.append(Unit(label=unit.label, section=unit.section,
                            lines=[("\n".join(buf), p_start)],
                            labels=list(unit.labels) or ([unit.label] if unit.label else [])))
            buf, buf_words = [], 0

    for para in paragraphs:
        if " | " in para:  # keep table rows grouped loosely
            if buf_words + count_words(para) > limit:
                spill()
            buf.append(para)
            buf_words += count_words(para)
            continue
        for sent in util.sentence_split(para) or [para]:
            w = count_words(sent)
            if buf and buf_words + w > limit:
                spill()
            buf.append(sent)
            buf_words += w
    spill()
    for u in out:
        u.lines = [(t, p_start) for t, _ in u.lines]
    return out or [unit]


def pack(units: list[Unit]) -> list[Unit]:
    """Greedy merge of adjacent units, flushing oversized leaves."""
    chunks: list[Unit] = []
    cur = Unit()
    for unit in units:
        pieces = _split_oversized(unit, HARD_MAX_WORDS) if unit.words() > HARD_MAX_WORDS else [unit]
        for piece in pieces:
            if piece.section != cur.section and cur.lines:
                if cur.text().strip():
                    chunks.append(cur)
                cur = Unit(section=piece.section)
            if cur.lines and cur.words() + piece.words() > MAX_WORDS and (
                    cur.words() >= MIN_WORDS or cur.words() + piece.words() > HARD_MAX_WORDS):
                chunks.append(cur)
                cur = Unit(section=piece.section)
            if not cur.lines:
                cur.section = piece.section
            if not cur.label and piece.label:
                cur.label = piece.label  # chunk is cited by its opening paragraph
            for lab in piece.labels or ([piece.label] if piece.label else []):
                if not cur.labels or cur.labels[-1] != lab:
                    cur.labels.append(lab)
            cur.lines.extend(piece.lines)
    if cur.text().strip():
        chunks.append(cur)
    return chunks


def build_chunks(doc_id: str, pages: list[dict]) -> list[dict]:
    """Parsed pages -> chunk records (doc-level metadata is added by the caller)."""
    lines = prepare_lines(pages)
    units = build_units(lines)
    packed = pack(units)
    out: list[dict] = []
    for i, unit in enumerate(packed):
        text = unit.text().strip()
        if not text:
            continue
        p0, p1 = unit.pages()
        out.append({
            "chunk_id": f"{doc_id}::{i:04d}",
            "doc_id": doc_id,
            "ordinal": i,
            "text": text,
            "n_words": count_words(text),
            "section": unit.section,
            "para": unit.label,
            "para_end": unit.labels[-1] if unit.labels else unit.label,
            "page_start": p0,
            "page_end": p1,
            "text_hash": util.sha256_text(text)[:16],
        })
    return out
