"""Reliable PDF text extraction and cleaning for legal documents.

Responsible for exactly one job: turn a PDF's bytes into a clean, per-page
list of text, while detecting the failure modes that a legal-corpus
ingestion pipeline has to refuse loudly rather than silently mis-index
(encrypted PDFs, scanned/image-only PDFs, empty PDFs, corrupt files).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class PDFIngestionError(Exception):
    """Base class for all recoverable PDF ingestion failures."""


class EncryptedPDFError(PDFIngestionError):
    pass


class ScannedPDFError(PDFIngestionError):
    pass


class EmptyPDFError(PDFIngestionError):
    pass


class ExtractionError(PDFIngestionError):
    pass


class UnsupportedFormatError(PDFIngestionError):
    pass


# A page is treated as "likely scanned" if it has fewer than this many
# extractable characters. Real legal-text pages run into the thousands.
MIN_CHARS_PER_PAGE = 25

# If this fraction (or more) of pages look scanned, refuse the whole file
# rather than silently indexing a document that is mostly blank.
SCANNED_PAGE_RATIO_THRESHOLD = 0.6

# A line repeated on at least this fraction of pages is treated as a
# running header/footer (title banners, "GOVERNMENT OF INDIA", etc.).
REPEATED_LINE_RATIO_THRESHOLD = 0.4

_PAGE_NUMBER_LINE = re.compile(r"^\s*(page\s*)?[-–—]?\s*\d{1,4}\s*[-–—]?\s*$", re.IGNORECASE)
_WHITESPACE = re.compile(r"[ \t]+")
_HYPHEN_LINEBREAK = re.compile(r"(\w)-\n(\w)")
_MULTI_BLANK = re.compile(r"\n{3,}")


@dataclass
class ParsedPDF:
    filename: str
    page_texts: list[str]  # cleaned text, one entry per non-blank page (1:1 with page_numbers)
    page_numbers: list[int]  # original 1-indexed page number for each entry in page_texts
    full_text: str  # cleaned pages joined with form-feed markers, used for structure detection
    num_pages_total: int
    num_pages_used: int
    removed_boilerplate_lines: list[str]


def _open_reader(file_bytes: bytes, filename: str) -> PdfReader:
    try:
        reader = PdfReader(__import__("io").BytesIO(file_bytes))
    except PdfReadError as exc:
        raise ExtractionError(f"'{filename}' could not be parsed as a PDF: {exc}") from exc
    except Exception as exc:  # pragma: no cover - defensive, e.g. non-PDF bytes
        raise UnsupportedFormatError(f"'{filename}' is not a readable PDF file: {exc}") from exc

    if reader.is_encrypted:
        # Some "encrypted" PDFs use an empty owner password and are readable
        # once decrypted; try that before giving up.
        try:
            if reader.decrypt("") == 0:
                raise EncryptedPDFError(
                    f"'{filename}' is password-protected. Remove the password and re-upload."
                )
        except EncryptedPDFError:
            raise
        except Exception as exc:
            raise EncryptedPDFError(
                f"'{filename}' is encrypted and could not be opened: {exc}"
            ) from exc
    return reader


def _raw_pages(reader: PdfReader, filename: str) -> list[str]:
    pages: list[str] = []
    for index, page in enumerate(reader.pages):
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            # A single malformed page shouldn't sink the whole document;
            # treat it as blank and let the scanned/empty checks handle it.
            pages.append("")
    if not pages:
        raise EmptyPDFError(f"'{filename}' has no pages.")
    return pages


def _clean_page_text(text: str) -> str:
    text = _HYPHEN_LINEBREAK.sub(r"\1\2", text)  # rejoin hyphenated line-wrap words
    text = _WHITESPACE.sub(" ", text)
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln and not _PAGE_NUMBER_LINE.match(ln)]
    return "\n".join(lines)


def _detect_boilerplate(cleaned_pages: list[str]) -> list[str]:
    """Find lines that repeat across many pages: running headers/footers,
    repeated document titles, watermark text."""
    if len(cleaned_pages) < 3:
        return []
    line_page_counts: Counter[str] = Counter()
    for page in cleaned_pages:
        for line in {ln.strip() for ln in page.split("\n") if ln.strip()}:
            if 0 < len(line) <= 120:  # ignore accidental full-paragraph "repeats"
                line_page_counts[line] += 1
    threshold = max(3, int(len(cleaned_pages) * REPEATED_LINE_RATIO_THRESHOLD))
    return [line for line, count in line_page_counts.items() if count >= threshold]


def parse_pdf(file_bytes: bytes, filename: str) -> ParsedPDF:
    """Extract, clean, and validate a legal PDF. Raises a PDFIngestionError
    subclass for any condition the caller should surface to the user."""
    if not filename.lower().endswith(".pdf"):
        raise UnsupportedFormatError(f"'{filename}' is not a .pdf file.")
    if not file_bytes:
        raise EmptyPDFError(f"'{filename}' is an empty upload.")

    reader = _open_reader(file_bytes, filename)
    raw_pages = _raw_pages(reader, filename)
    num_pages_total = len(raw_pages)

    cleaned_pages = [_clean_page_text(p) for p in raw_pages]
    scanned_like = sum(1 for p in cleaned_pages if len(p) < MIN_CHARS_PER_PAGE)
    if num_pages_total and (scanned_like / num_pages_total) >= SCANNED_PAGE_RATIO_THRESHOLD:
        raise ScannedPDFError(
            f"'{filename}' appears to be a scanned/image-only PDF "
            f"({scanned_like}/{num_pages_total} pages had no extractable text). "
            "Run OCR (e.g. ocrmypdf) before ingesting."
        )

    boilerplate = _detect_boilerplate(cleaned_pages)
    boilerplate_set = set(boilerplate)

    page_texts: list[str] = []
    page_numbers: list[int] = []
    for page_index, page in enumerate(cleaned_pages, start=1):
        if boilerplate_set:
            lines = [ln for ln in page.split("\n") if ln.strip() not in boilerplate_set]
            page = "\n".join(lines)
        page = page.strip()
        if len(page) < MIN_CHARS_PER_PAGE:
            continue  # blank page or boilerplate-only page
        page_texts.append(page)
        page_numbers.append(page_index)

    if not page_texts:
        raise EmptyPDFError(
            f"'{filename}' produced no usable text after cleaning "
            "(likely blank pages only or an image-only document)."
        )

    full_text = _MULTI_BLANK.sub("\n\n", "\f".join(page_texts))

    return ParsedPDF(
        filename=filename,
        page_texts=page_texts,
        page_numbers=page_numbers,
        full_text=full_text,
        num_pages_total=num_pages_total,
        num_pages_used=len(page_texts),
        removed_boilerplate_lines=boilerplate,
    )
