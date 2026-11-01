"""DJVU: извлечение скрытого текстового слоя через djvutxt."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

from book_expert.ingestion.document import (
    DocumentError,
    ExtractedDocument,
    ExtractedPage,
)
from book_expert.ingestion.pdf import PdfParseError

DJVU_TIMEOUT_S = 180


class DjvuToolMissingError(DocumentError):
    """Не установлен djvutxt (пакет djvulibre-bin)."""


class DjvuNoTextError(DocumentError):
    """В DJVU нет текстового слоя."""


def extract_djvu(path: str | Path) -> ExtractedDocument:
    djvu_path = Path(path).expanduser()
    if not djvu_path.is_file():
        raise FileNotFoundError(djvu_path)
    with djvu_path.open("rb") as handle:
        if handle.read(8) != b"AT&TFORM":
            raise PdfParseError("not a DJVU document")
    tool = shutil.which("djvutxt")
    if tool is None:
        raise DjvuToolMissingError(
            "djvutxt is not installed; install djvulibre-bin to read DJVU"
        )
    try:
        result = subprocess.run(
            [tool, str(djvu_path)],
            capture_output=True,
            timeout=DJVU_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise PdfParseError("cannot run djvutxt") from exc
    if result.returncode != 0:
        raise PdfParseError("djvutxt failed to extract DJVU text")
    text = result.stdout.decode("utf-8", errors="replace")
    parts = text.split("\f")
    if parts and not parts[-1].strip():
        parts.pop()
    pages = tuple(
        ExtractedPage(
            pdf_page_index=index,
            text=part.strip(),
            printed_page_label=None,
            char_count=len(part.strip()),
            block_count=len(part.strip().splitlines()),
        )
        for index, part in enumerate(parts)
    )
    if not any(not page.is_empty for page in pages):
        raise DjvuNoTextError("DJVU has no text layer; OCR is not enabled")
    return ExtractedDocument(
        path=djvu_path.resolve(),
        content_hash=_sha256(djvu_path),
        page_count=len(pages),
        pages=pages,
        format="djvu",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["DjvuNoTextError", "DjvuToolMissingError", "extract_djvu"]
