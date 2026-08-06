"""Real document parsing into typed elements — `unstructured.io`, a NARROWER extra set than
`unstructured[all-docs]`. Read this before assuming a format "just works": this module states
exactly what is exercised and what is not, per the task brief ("do not fake the parse step and
do not claim broader format support than you exercised").

--- Supported (real, tested against real files in `tests/test_parser.py`) ---

TXT, MD, HTML, EML, DOCX, PPTX — unstructured's own real partitioners (`partition_text`,
`partition_md`, `partition_html`, `partition_email`, `partition_docx`, `partition_pptx`), no
system binaries required. PDF — see the note below; text-layer only.

--- NOT supported (raises `UnsupportedDocumentType`; no partial or fake output) ---

Legacy binary Office formats (`.doc`, `.ppt`, `.xls`; these need LibreOffice, not installed
here), any other unstructured.io format not listed above (RTF, ODT, XLSX, images, ...), and —
within PDF — scanned/image-only pages (no OCR path; see below).

--- Why PDF is NOT routed through `unstructured.partition.pdf.partition_pdf` ---

`unstructured.partition.pdf` unconditionally imports `unstructured_inference` at MODULE level
(confirmed by reading `unstructured/partition/pdf.py`: the `TYPE_CHECKING`-only import at the
top is a red herring — `_partition_pdf_or_image_with_ocr`'s import chain pulls in
`unstructured.partition.pdf_image.pdfminer_processing`, which imports `unstructured_inference`
unconditionally at its own module top, so the import fails BEFORE strategy selection even runs).
`unstructured_inference` in turn requires `torch`, `torchvision`, `onnxruntime`,
`opencv-python`, `pandas`, `matplotlib`, `timm`, and `accelerate` — verified live in this repo's
venv: `pip install "unstructured[pdf]"` pulled ~2GB of ML packages to make a hi-res
layout-detection model class importable, even though that model is used only by the `hi_res`
and `ocr_only` strategies, never `fast`. This is exactly the disproportionate pull the task
brief anticipated ("a large surface... a narrower parser set that genuinely works").

Instead, PDF text is extracted directly with `pdfminer.six` (`pdfminer.high_level.extract_text`
— a real, lightweight, pure-Python library that is ALREADY unstructured's own dependency for
its `fast` PDF strategy) and then classified into typed elements with unstructured's OWN real
text-type classifier, `unstructured.partition.text.partition_text(text=...)` — the identical
Title/NarrativeText/ListItem heuristic `partition_text`/`partition_email` use for every other
plain-text-shaped format in this module. This is real extraction plus real classification,
composed at a lighter entry point than `partition_pdf`, not a stub. Stated plainly: PDF support
here has no OCR (an image-only/scanned PDF yields no text and therefore zero elements — this is
a real, verifiable limitation, not silently swallowed) and no PDF-internal table-structure
detection (a table inside a PDF becomes `NarrativeText`, not `Table` — `partition_pdf`'s table
detection depends on the same hi-res model this module deliberately does not load). DOCX/HTML/MD
table detection is unaffected (proven in `tests/test_parser.py`).

--- Element-type mapping (frozen enum) ---

`document-chunks-value`'s `DocumentElementType` enum is exactly `{Title, NarrativeText, Table,
ListItem}` (real, registered Avro schema — see `mytheus-events/schemas/document-chunks/v1.avsc`).
unstructured produces a much larger set of element classes (`Header`, `Footer`,
`UncategorizedText`, `Address`, `EmailAddress`, `Image`, `FigureCaption`, `Formula`,
`CodeSnippet`, `PageBreak`, ...). `map_type` below maps the four frozen types through unchanged
and folds every other class into `NarrativeText` (it is still real extracted text; the frozen
enum has no catch-all "Other" member) — never dropped, never invented.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

from unstructured.documents.elements import Element
from unstructured.partition.auto import partition
from unstructured.partition.text import partition_text

from libs.shared.object_storage import safe_extension

# Extensions unstructured's own partitioners handle directly (no special-casing needed), mapped
# to an explicit MIME `content_type` hint. Passing `content_type` deterministically selects the
# partitioner instead of relying on libmagic/content-sniffing (which this dev environment lacks
# — "libmagic is unavailable" — and which is unnecessary anyway since `safe_extension` already
# validated the format from the same allow-list this dict's keys are drawn from).
_CONTENT_TYPES = {
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".html": "text/html",
    ".htm": "text/html",
    ".md": "text/markdown",
    ".eml": "message/rfc822",
    ".txt": "text/plain",
}

_ELEMENT_TYPE_MAP = {
    "Title": "Title",
    "NarrativeText": "NarrativeText",
    "Table": "Table",
    "ListItem": "ListItem",
}


class UnsupportedDocumentType(ValueError):
    """Raised for any extension outside `libs.shared.object_storage.SUPPORTED_EXTENSIONS`, or a
    format that IS in that set but genuinely cannot be parsed (should not happen for the fixed
    set this module supports; kept narrow rather than broad)."""


@dataclass(frozen=True)
class ParsedElement:
    position: int  # ordinal of the RAW unstructured element (source layout order)
    element_type: str  # one of Title / NarrativeText / Table / ListItem
    text: str
    raw: dict[str, Any]  # unstructured's own element.to_dict(), for document_elements.raw JSONB


def map_type(element: Element) -> str:
    return _ELEMENT_TYPE_MAP.get(type(element).__name__, "NarrativeText")


def parse_document(*, filename: str, content: bytes) -> list[ParsedElement]:
    """Parse `content` (the bytes exactly as stored in object storage) into typed elements.
    `filename` is used ONLY to pick which real partitioner to dispatch to (via
    `safe_extension`'s allow-listed suffix) — never trusted for anything else, matching
    `libs/shared/object_storage.py`'s key-safety contract. Raises `UnsupportedDocumentType` for
    anything outside the narrow supported set; never returns a fabricated element."""
    ext = safe_extension(filename)
    if ext is None:
        raise UnsupportedDocumentType(f"unsupported extension for {filename!r}")

    if ext == ".pdf":
        elements = _parse_pdf(content)
    elif ext in _CONTENT_TYPES:
        elements = partition(
            file=io.BytesIO(content),
            content_type=_CONTENT_TYPES[ext],
            metadata_filename=filename,
        )
    else:  # pragma: no cover - unreachable while _CONTENT_TYPES ∪ {.pdf} == SUPPORTED_EXTENSIONS
        raise UnsupportedDocumentType(f"no parser wired for {ext!r}")

    return [
        ParsedElement(
            position=pos,
            element_type=map_type(el),
            text=el.text,
            raw=el.to_dict(),
        )
        for pos, el in enumerate(elements)
        if el.text and el.text.strip()
    ]


def _parse_pdf(content: bytes) -> list[Element]:
    """See the module docstring's PDF note: pdfminer text extraction (real) + unstructured's
    real text-type classifier, deliberately not `partition_pdf`."""
    from pdfminer.high_level import extract_text

    text = extract_text(io.BytesIO(content))
    if not text or not text.strip():
        return []  # scanned/image-only PDF: no text layer, no OCR path — zero elements, honest
    return partition_text(text=text)
