"""Layout-aware PDF extraction on a generated PDF."""
import fitz

from regintel import extract


def _make_pdf(tmp_path, body: str, pages: int = 1):
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page()
        page.insert_text((72, 90), body, fontsize=11)
    path = tmp_path / "doc.pdf"
    doc.save(path)
    doc.close()
    return path


def test_extract_simple(tmp_path):
    body = ("CHAPTER I\n1. Scope. Every entity shall maintain records for five years. "
            "The records shall be retrievable at all times.")
    pages = extract.extract_pdf(_make_pdf(tmp_path, body))
    assert len(pages) == 1
    assert "maintain records" in pages[0]["text"]
    assert pages[0]["page"] == 1


def test_drops_repeated_footer(tmp_path):
    doc = fitz.open()
    for n in range(4):
        page = doc.new_page()
        page.insert_text((72, 90), f"Paragraph {n}. The entity shall maintain records of all transactions.",
                         fontsize=11)
        page.insert_text((72, 780), "Regulated Entities: Compliance Cell\nPage {}/4".format(n + 1),
                         fontsize=7)
    path = tmp_path / "foot.pdf"
    doc.save(path)
    doc.close()
    pages = extract.extract_pdf(path)
    joined = " ".join(p["text"] for p in pages)
    assert "Compliance Cell" not in joined
    assert "Paragraph 3" in joined


def test_dehyphenate():
    assert extract._dehyphenate("regu-\nlation and com-\npliance") == "regulation and compliance"
