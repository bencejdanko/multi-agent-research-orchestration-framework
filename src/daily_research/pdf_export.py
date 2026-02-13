"""PDF export — convert Markdown reports to PDF via pandoc + weasyprint."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


def markdown_to_pdf(md_path: Path, pdf_path: Path | None = None) -> Path:
    """Convert a Markdown file to PDF using pandoc with weasyprint.

    Parameters
    ----------
    md_path:
        Path to the source ``.md`` file.
    pdf_path:
        Destination path for the PDF.  Defaults to the same directory and
        stem as *md_path* with a ``.pdf`` extension.

    Returns
    -------
    Path
        The path to the generated PDF file.

    Raises
    ------
    RuntimeError
        If pandoc exits with a non-zero status.
    FileNotFoundError
        If pandoc is not installed.
    """
    if pdf_path is None:
        pdf_path = md_path.with_suffix(".pdf")

    cmd = [
        "pandoc",
        str(md_path),
        "-o",
        str(pdf_path),
        "--pdf-engine=weasyprint",
        "--metadata",
        f"title={md_path.stem}",
    ]

    log.info("Running: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            "pandoc is not installed or not on PATH. "
            "Install it with: sudo apt install pandoc"
        )

    if result.returncode != 0:
        log.error("pandoc stderr: %s", result.stderr)
        raise RuntimeError(f"pandoc failed (exit {result.returncode}): {result.stderr}")

    log.info("PDF generated → %s", pdf_path)
    return pdf_path
