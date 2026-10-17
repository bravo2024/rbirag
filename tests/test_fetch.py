"""Notification page -> PDF link + reference extraction (offline fixture)."""
from conftest import FIXTURES
from regintel import fetch


def test_find_pdf_and_reference():
    html = (FIXTURES / "notification_page.html").read_text(encoding="utf-8")
    pdf_url, ref = fetch.find_pdf("https://www.rbi.org.in/scripts/NotificationUser.aspx?Id=13739", html=html)
    assert pdf_url and pdf_url.startswith("https://rbidocs.rbi.org.in/")
    assert pdf_url.lower().endswith(".pdf")
    assert ref and ref.startswith("RBI/2026-27/")


def test_pdf_head_ref(tmp_path):
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "RBI/2025-26/100\nDOR.RRC.REC.302 test head", fontsize=11)
    path = tmp_path / "t.pdf"
    doc.save(path)
    doc.close()
    assert fetch.pdf_head_ref(path) == "RBI/2025-26/100"
