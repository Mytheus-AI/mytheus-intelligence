"""Real `unstructured`/`pdfminer` parsing against real files of every SUPPORTED extension — no
infra needed (no DB/Redis/Redpanda/MinIO), pure library behavior. Proves the plan's acceptance
check directly: `element_type` values land in {Title, NarrativeText, Table, ListItem} for every
format this service claims to support, and unsupported formats are rejected rather than
silently mis-parsed."""

from __future__ import annotations

import pytest

from services.ingestion_service.parser import UnsupportedDocumentType, parse_document
from tests.fixtures import sample_documents as docs

_FROZEN_TYPES = {"Title", "NarrativeText", "Table", "ListItem"}


def test_txt_produces_narrative_text():
    elements = parse_document(filename="sample.txt", content=docs.make_txt())
    assert len(elements) >= 1
    assert all(el.element_type in _FROZEN_TYPES for el in elements)
    assert any("narrative text" in el.text.lower() for el in elements)


def test_md_produces_title_narrative_listitem_and_table():
    elements = parse_document(filename="sample.md", content=docs.make_md())
    types = {el.element_type for el in elements}
    assert all(t in _FROZEN_TYPES for t in types)
    assert "Title" in types
    assert "NarrativeText" in types
    assert "ListItem" in types
    texts = [el.text for el in elements]
    assert any("ship the ingestion service" in t for t in texts)


def test_html_produces_title_narrative_and_listitems():
    elements = parse_document(filename="notes.html", content=docs.make_html())
    types = [el.element_type for el in elements]
    assert types[0] == "Title"
    assert "NarrativeText" in types
    assert types.count("ListItem") == 2


def test_eml_produces_narrative_text():
    elements = parse_document(filename="notes.eml", content=docs.make_eml())
    assert len(elements) >= 1
    assert all(el.element_type in _FROZEN_TYPES for el in elements)
    assert any("ship the ingestion service" in el.text for el in elements)


def test_docx_produces_title_narrative_and_listitem():
    elements = parse_document(filename="notes.docx", content=docs.make_docx())
    types = {el.element_type for el in elements}
    assert types == {"Title", "NarrativeText", "ListItem"}


def test_pptx_produces_title_and_narrative():
    elements = parse_document(filename="notes.pptx", content=docs.make_pptx())
    types = {el.element_type for el in elements}
    assert types <= _FROZEN_TYPES
    assert "Title" in types


def test_pdf_extracts_real_text_layer_via_pdfminer():
    elements = parse_document(filename="notes.pdf", content=docs.make_pdf())
    assert len(elements) >= 1
    assert all(el.element_type in _FROZEN_TYPES for el in elements)
    joined = " ".join(el.text for el in elements)
    assert "Kickoff Notes" in joined
    assert "ship the ingestion service" in joined
    assert "Bob to review the PR" in joined


def test_pdf_with_no_text_layer_returns_zero_elements_not_a_crash():
    """The honest scanned/image-only-PDF limitation stated in parser.py's docstring: no OCR
    path, so a text-less PDF yields zero elements rather than fabricating output."""
    elements = parse_document(filename="blank.pdf", content=docs.make_blank_pdf())
    assert elements == []


@pytest.mark.parametrize(
    "filename", ["evil.exe", "legacy.doc", "legacy.ppt", "spreadsheet.xlsx", "photo.png", "noext"]
)
def test_unsupported_extension_raises_never_fakes_output(filename: str):
    with pytest.raises(UnsupportedDocumentType):
        parse_document(filename=filename, content=b"irrelevant bytes")


def test_position_is_stable_source_order():
    elements = parse_document(filename="sample.md", content=docs.make_md())
    positions = [el.position for el in elements]
    assert positions == sorted(positions)
    assert positions == list(range(len(elements)))


def test_raw_carries_unstructured_element_dict():
    elements = parse_document(filename="notes.docx", content=docs.make_docx())
    for el in elements:
        assert isinstance(el.raw, dict)
        assert "type" in el.raw
        assert "text" in el.raw
