"""Нативный FB2-парсер: секции → виртуальные страницы."""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

from book_expert.ingestion.document import ExtractedDocument, ExtractedPage
from book_expert.ingestion.pdf import PdfParseError

MAX_FB2_BYTES = 64 * 1024 * 1024


def extract_fb2(path: str | Path) -> ExtractedDocument:
    fb2_path = Path(path).expanduser()
    if not fb2_path.is_file():
        raise FileNotFoundError(fb2_path)
    data = fb2_path.read_bytes()
    if len(data) > MAX_FB2_BYTES:
        raise PdfParseError("FB2 is larger than 64 MB")
    root = _parse(data)
    if _tag(root) != "FictionBook":
        raise PdfParseError("not an FB2 document")

    title, author, language, annotation = _metadata(root)
    pages: list[ExtractedPage] = []
    for body in _children(root, "body"):
        sections = _children(body, "section")
        if sections:
            for section in sections:
                _walk_section(section, [], pages)
        else:
            text = _block_text(body)
            if text.strip():
                pages.append(_page(len(pages), text, None, body))
    if not any(not page.is_empty for page in pages):
        raise PdfParseError("FB2 has no extractable text")
    metadata: dict[str, str] = {}
    if language:
        metadata["language"] = language
    if annotation:
        metadata["annotation"] = annotation
    return ExtractedDocument(
        path=fb2_path.resolve(),
        content_hash=_sha256(fb2_path),
        page_count=len(pages),
        pages=tuple(pages),
        title=title,
        author=author,
        metadata=metadata or None,
        format="fb2",
    )


def _walk_section(
    section: ET.Element, prefix: list[str], pages: list[ExtractedPage]
) -> None:
    title = _section_title(section)
    path = " > ".join(prefix + ([title] if title else []))
    text = _block_text(section)
    if text.strip() or title:
        page_text = "\n".join(part for part in (title, text) if part)
        pages.append(_page(len(pages), page_text, path or None, section))
    for child in _children(section, "section"):
        _walk_section(child, prefix + ([title] if title else []), pages)


def _page(
    index: int, text: str, label: str | None, element: ET.Element
) -> ExtractedPage:
    return ExtractedPage(
        pdf_page_index=index,
        text=text,
        printed_page_label=label,
        char_count=len(text),
        block_count=len(_children(element, "p")),
    )


def _section_title(section: ET.Element) -> str | None:
    titles = _children(section, "title")
    if not titles:
        return None
    text = " ".join(_text(child) for child in titles[0] if _text(child)).strip()
    return text or None


def _block_text(element: ET.Element) -> str:
    parts: list[str] = []
    for child in element:
        tag = _tag(child)
        if tag in {"section", "title"}:
            continue
        if tag in {"p", "subtitle", "v", "text-author", "td", "th"}:
            text = _text(child)
            if text:
                parts.append(text)
        elif tag in {"poem", "stanza", "cite", "epigraph", "table", "annotation"}:
            nested = _block_text(child)
            if nested:
                parts.append(nested)
    return "\n".join(parts)


def _metadata(
    root: ET.Element,
) -> tuple[str | None, str | None, str | None, str | None]:
    title_info = None
    for description in _children(root, "description"):
        for name in ("title-info",):
            found = _children(description, name)
            if found:
                title_info = found[0]
                break
    if title_info is None:
        return None, None, None, None

    def first(name: str) -> str | None:
        nodes = _children(title_info, name)
        if not nodes:
            return None
        value = _text(nodes[0]).strip()
        return value or None

    title = first("book-title")
    language = first("lang")
    annotation = first("annotation")
    authors: list[str] = []
    for author_node in _children(title_info, "author"):
        parts = [
            _text(child).strip()
            for child in author_node
            if _tag(child) in {"first-name", "middle-name", "last-name"}
        ]
        name = " ".join(part for part in parts if part)
        if name:
            authors.append(name)
    return title, ", ".join(authors) or None, language, annotation


def _parse(data: bytes) -> ET.Element:
    probe = data[:8192]
    if probe.startswith((b"\xff\xfe", b"\xfe\xff")):
        probe_text = probe.decode("utf-16", errors="ignore")
    elif probe[:2] in {b"<\x00", b"\x00<"}:
        probe_text = probe.decode(
            "utf-16-le" if probe[:2] == b"<\x00" else "utf-16-be", errors="ignore"
        )
    else:
        probe_text = probe.decode("utf-8", errors="ignore")
    if "<!ENTITY" in probe_text:
        raise PdfParseError("FB2 entity declarations are not allowed")
    try:
        return ET.fromstring(data)
    except ET.ParseError:
        try:
            return ET.fromstring(data.decode("cp1251"))
        except (UnicodeDecodeError, ET.ParseError) as exc:
            raise PdfParseError("cannot parse FB2 XML") from exc


def _children(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in element if _tag(child) == name]


def _tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _text(element: ET.Element) -> str:
    return "".join(element.itertext()).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["extract_fb2"]
