"""PDF acquisition and text extraction for arXiv papers."""

from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path

import fitz  # PyMuPDF
import httpx

log = logging.getLogger(__name__)

# Common section headings in academic papers (case-insensitive patterns)
_SECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("abstract", re.compile(r"^(?:abstract)\s*$", re.IGNORECASE)),
    ("introduction", re.compile(r"^(?:\d+[\.\)]?\s*)?introduction\s*$", re.IGNORECASE)),
    ("related_work", re.compile(r"^(?:\d+[\.\)]?\s*)?(?:related\s+work|background|prior\s+work|literature\s+review)\s*$", re.IGNORECASE)),
    ("method", re.compile(r"^(?:\d+[\.\)]?\s*)?(?:method(?:ology|s)?|approach|model|framework|proposed\s+method|our\s+approach)\s*$", re.IGNORECASE)),
    ("experiments", re.compile(r"^(?:\d+[\.\)]?\s*)?(?:experiment(?:s|al)?(?:\s+(?:setup|results))?|evaluation)\s*$", re.IGNORECASE)),
    ("results", re.compile(r"^(?:\d+[\.\)]?\s*)?(?:results?(?:\s+and\s+discussion)?|main\s+results)\s*$", re.IGNORECASE)),
    ("discussion", re.compile(r"^(?:\d+[\.\)]?\s*)?discussion\s*$", re.IGNORECASE)),
    ("limitations", re.compile(r"^(?:\d+[\.\)]?\s*)?(?:limitations?|limitations?\s+and\s+future\s+work)\s*$", re.IGNORECASE)),
    ("conclusion", re.compile(r"^(?:\d+[\.\)]?\s*)?(?:conclusion(?:s)?|summary(?:\s+and\s+(?:conclusion|future\s+work))?)\s*$", re.IGNORECASE)),
    ("references", re.compile(r"^(?:references|bibliography)\s*$", re.IGNORECASE)),
    ("appendix", re.compile(r"^(?:appendi(?:x|ces))\s*$", re.IGNORECASE)),
]


def download_pdf(url: str, dest_dir: Path | None = None) -> Path | None:
    """Download a PDF from a URL and return the local path.

    Uses a temporary directory if *dest_dir* is not supplied.
    """
    if not url:
        return None

    try:
        dest_dir = dest_dir or Path(tempfile.mkdtemp(prefix="arxiv_pdf_"))
        # Build a filename from the URL
        filename = url.rsplit("/", 1)[-1]
        if not filename.endswith(".pdf"):
            filename += ".pdf"
        dest = dest_dir / filename

        if dest.exists():
            log.debug("PDF already cached: %s", dest)
            return dest

        log.info("Downloading PDF: %s", url)
        with httpx.stream("GET", url, follow_redirects=True, timeout=60) as resp:
            resp.raise_for_status()
            with dest.open("wb") as f:
                for chunk in resp.iter_bytes(chunk_size=8192):
                    f.write(chunk)

        log.info("PDF saved: %s (%.1f KB)", dest.name, dest.stat().st_size / 1024)
        return dest

    except Exception:
        log.exception("Failed to download PDF: %s", url)
        return None


def extract_text(pdf_path: Path) -> str:
    """Extract all text from a PDF using PyMuPDF."""
    try:
        doc = fitz.open(str(pdf_path))
        pages: list[str] = []
        for page in doc:
            pages.append(page.get_text())
        doc.close()
        return "\n\n".join(pages)
    except Exception:
        log.exception("Failed to extract text from: %s", pdf_path)
        return ""


def _classify_line(line: str) -> str | None:
    """Check if a line looks like a section heading.  Returns section name or None."""
    stripped = line.strip()
    if not stripped or len(stripped) > 100:
        return None
    for section_name, pattern in _SECTION_PATTERNS:
        if pattern.match(stripped):
            return section_name
    return None


def extract_sections(full_text: str) -> dict[str, str]:
    """Split full paper text into named sections.

    Returns a dict mapping section name → section content.
    Falls back to ``{'full': <entire text>}`` if no headings are detected.
    """
    lines = full_text.split("\n")
    sections: dict[str, list[str]] = {}
    current_section = "preamble"
    sections[current_section] = []

    for line in lines:
        heading = _classify_line(line)
        if heading:
            current_section = heading
            if current_section not in sections:
                sections[current_section] = []
        else:
            sections.setdefault(current_section, []).append(line)

    result: dict[str, str] = {}
    for name, content_lines in sections.items():
        text = "\n".join(content_lines).strip()
        if text:
            result[name] = text

    # If we only found preamble, return as 'full'
    if list(result.keys()) == ["preamble"]:
        return {"full": result["preamble"]}

    return result


def extract_references_text(full_text: str) -> str:
    """Extract the references section from full text."""
    sections = extract_sections(full_text)
    return sections.get("references", "")


def find_arxiv_ids_in_text(text: str) -> list[str]:
    """Find all arXiv IDs mentioned in the text.

    Matches patterns like:
      - arXiv:2301.12345
      - arxiv.org/abs/2301.12345
      - 2301.12345 (four-digit year-month pattern)
    """
    patterns = [
        r"arXiv[:\s]+(\d{4}\.\d{4,5})",
        r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})",
        # Standalone arXiv ID pattern (be conservative)
        r"(?<!\d)(\d{4}\.\d{5})(?!\d)",
    ]
    ids: set[str] = set()
    for pat in patterns:
        ids.update(re.findall(pat, text, re.IGNORECASE))
    return sorted(ids)


def download_and_extract(
    pdf_url: str,
    *,
    cache_dir: Path | None = None,
) -> tuple[str, dict[str, str]]:
    """Download a PDF, extract full text and sections.

    Returns (full_text, sections_dict).
    """
    pdf_path = download_pdf(pdf_url, dest_dir=cache_dir)
    if not pdf_path:
        return "", {}

    full_text = extract_text(pdf_path)
    if not full_text:
        return "", {}

    sections = extract_sections(full_text)
    return full_text, sections
