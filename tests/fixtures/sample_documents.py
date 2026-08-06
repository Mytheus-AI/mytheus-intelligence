"""Real sample documents for `tests/test_parser.py` and the pipeline/route/consumer tests. Every
byte string here is a genuine, valid file of its format — parsed by the REAL library for that
format (python-docx, python-pptx, pdfminer via a hand-rolled minimal PDF), not a fixture that
merely has the right extension. No dependency was added purely to generate test fixtures
(`reportlab` etc.) — the PDF generator below is a minimal, self-contained PDF writer using only
the standard library, kept intentionally small since its only job is to produce a real,
pdfminer-readable single-page text PDF.
"""

from __future__ import annotations

import io


def make_txt() -> bytes:
    return b"Plain text ingestion smoke test.\nThis line should become narrative text.\n"


def make_md() -> bytes:
    return (
        b"# Project Kickoff\n\n"
        b"## Decisions\n\n"
        b"We decided to ship the ingestion service this sprint.\n\n"
        b"- Alice owns the parser\n"
        b"- Bob owns object storage\n"
    )


def make_html() -> bytes:
    return (
        b"<html><body>"
        b"<h1>Meeting Notes</h1>"
        b"<p>We decided to adopt MinIO for real object storage.</p>"
        b"<ul><li>Action: wire the bucket</li><li>Action: write tests</li></ul>"
        b"</body></html>"
    )


def make_eml() -> bytes:
    return (
        b"From: alice@example.com\r\n"
        b"To: bob@example.com\r\n"
        b"Subject: Kickoff notes\r\n"
        b"Date: Mon, 1 Jan 2024 00:00:00 +0000\r\n"
        b"Content-Type: text/plain\r\n"
        b"\r\n"
        b"We decided to ship the ingestion service this sprint.\r\n"
        b"Action item: Bob to review the PR.\r\n"
    )


def make_docx() -> bytes:
    from docx import Document

    buf = io.BytesIO()
    doc = Document()
    doc.add_heading("Kickoff", level=1)
    doc.add_paragraph("We decided to ship the ingestion service this sprint.")
    doc.add_paragraph("Bob owns the object storage integration.", style="List Bullet")
    doc.save(buf)
    return buf.getvalue()


def make_pptx() -> bytes:
    from pptx import Presentation

    buf = io.BytesIO()
    pres = Presentation()
    slide = pres.slides.add_slide(pres.slide_layouts[1])
    slide.shapes.title.text = "Kickoff"
    slide.placeholders[1].text = "We decided to ship the ingestion service this sprint."
    pres.save(buf)
    return buf.getvalue()


def make_pdf(lines: list[str] | None = None) -> bytes:
    """A minimal, hand-rolled, valid single-page PDF (stdlib only) carrying a real text layer —
    real bytes for pdfminer to really extract, not a stub."""
    if lines is None:
        lines = [
            "Kickoff Notes",
            "We decided to ship the ingestion service this sprint.",
            "Action item: Bob to review the PR.",
        ]

    content_lines = []
    y = 720
    for line in lines:
        esc = line.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        content_lines.append(f"BT /F1 12 Tf 72 {y} Td ({esc}) Tj ET")
        y -= 20
    content = "\n".join(content_lines).encode("latin-1")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> "
        b"/MediaBox [0 0 612 792] /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(content), content),
    ]

    buf = io.BytesIO()
    buf.write(b"%PDF-1.4\n")
    offsets = [0]
    for i, obj in enumerate(objects, start=1):
        offsets.append(buf.tell())
        buf.write(f"{i} 0 obj\n".encode())
        buf.write(obj)
        buf.write(b"\nendobj\n")
    xref_offset = buf.tell()
    n = len(objects) + 1
    buf.write(f"xref\n0 {n}\n".encode())
    buf.write(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        buf.write(f"{off:010d} 00000 n \n".encode())
    buf.write(b"trailer\n")
    buf.write(f"<< /Size {n} /Root 1 0 R >>\n".encode())
    buf.write(b"startxref\n")
    buf.write(f"{xref_offset}\n".encode())
    buf.write(b"%%EOF")
    return buf.getvalue()


def make_blank_pdf() -> bytes:
    """A real, valid PDF with NO text content (empty content stream) — exercises the scanned/
    image-only-PDF path (`parser.py`'s honest zero-elements behavior, not a crash)."""
    return make_pdf(lines=[])
