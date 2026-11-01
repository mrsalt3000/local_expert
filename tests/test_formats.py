"""Тесты поддержки EPUB, FB2 и DJVU."""

from __future__ import annotations

import hashlib
import io
import subprocess
import zipfile
from pathlib import Path
from typing import Any, Sequence

import pytest
from qdrant_client import QdrantClient

from book_expert.catalog import Catalog
from book_expert.config import Settings
from book_expert.ingestion.djvu import (
    DjvuNoTextError,
    DjvuToolMissingError,
    extract_djvu,
)
from book_expert.ingestion.chunking import chunk_document
from book_expert.ingestion.epub import extract_epub
from book_expert.ingestion.extract import detect_format, format_for_name
from book_expert.ingestion.fb2 import extract_fb2
from book_expert.ingestion.indexer import BookIndexer
from book_expert.ingestion.pdf import PdfParseError
from book_expert.models import BookStatus
from book_expert.providers import (
    ChatResult,
    EmbeddingResult,
    ProviderHealth,
)
from book_expert.web.library import LibraryService


class FakeProvider:
    name = "fake"
    chat_model = "fake-chat"
    embedding_model = "fake-embed"

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.calls = 0

    def chat(
        self,
        messages: Sequence[Any],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ChatResult:
        raise NotImplementedError

    def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        self.calls += 1
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            vectors.append(
                tuple(round(digest[index] / 255.0, 6) for index in range(self.dim))
            )
        return EmbeddingResult(vectors=tuple(vectors), model="fake-embed", dim=self.dim)

    def health(self) -> ProviderHealth:
        return ProviderHealth(ok=True, detail="fake")


class FakeIndexer:
    def index_pdf(self, path: str | Path, **kwargs: object):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def delete_book(self, book_id: str) -> bool:
        return True


def _settings(tmp_path: Path, *, dim: int = 4) -> Settings:
    return Settings(
        _env_file=None,
        workspace=tmp_path,
        data_dir=tmp_path / "data",
        embedding_dim=dim,
        chunk_size=400,
        chunk_overlap=40,
    )


def _write_epub(
    path: Path, chapters: list[str], headings: list[str] | None = None
) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED
        )
        archive.writestr(
            "META-INF/container.xml",
            '<?xml version="1.0"?><container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0"><rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>',
        )
        items = "".join(
            f'<item id="c{index}" href="ch{index}.xhtml" media-type="application/xhtml+xml"/>'
            for index in range(1, len(chapters) + 1)
        )
        refs = "".join(
            f'<itemref idref="c{index}"/>' for index in range(1, len(chapters) + 1)
        )
        archive.writestr(
            "OEBPS/content.opf",
            '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="id"><metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Synthetic EPUB</dc:title><dc:identifier id="id">epub-1</dc:identifier><dc:language>ru</dc:language></metadata><manifest>'
            + items
            + "</manifest><spine>"
            + refs
            + "</spine></package>",
        )
        for index, text in enumerate(chapters, start=1):
            heading = headings[index - 1] if headings else f"Глава {index}"
            archive.writestr(
                f"OEBPS/ch{index}.xhtml",
                f'<html xmlns="http://www.w3.org/1999/xhtml"><body><h1>{heading}</h1><p>{text}</p></body></html>',
            )
    return path


def _write_fb2(path: Path, encoding: str = "utf-8") -> Path:
    xml = """<?xml version="1.0" encoding="{encoding}"?>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">
<description><title-info><book-title>FB2 Check</book-title><lang>ru</lang>
<author><first-name>Иван</first-name><last-name>Петров</last-name></author>
</title-info></description>
<body>
<section><title><p>Глава 1</p></title><p>Первый абзац про поиск.</p>
<section><title><p>Раздел 1.1</p></title><p>Вложенный абзац.</p></section></section>
<section><title><p>Глава 2</p></title><p>Второй абзац.</p></section>
</body></FictionBook>""".format(encoding=encoding)
    path.write_bytes(xml.encode(encoding))
    return path


def _catalog(tmp_path: Path) -> Catalog:
    instance = Catalog(tmp_path / "data" / "artifacts" / "catalog.sqlite3")
    instance.initialize()
    return instance


def _indexer(tmp_path: Path, catalog: Catalog, client: QdrantClient) -> BookIndexer:
    return BookIndexer(
        _settings(tmp_path),
        FakeProvider(),
        catalog,
        client=client,
        batch_size=2,
    )


@pytest.mark.parametrize(
    ("raw", "fmt"),
    [
        ("book.pdf", "pdf"),
        ("book.epub", "epub"),
        ("book.fb2", "fb2"),
        ("book.djvu", "djvu"),
        ("BOOK.EPUB", "epub"),
    ],
)
def test_format_for_name(raw: str, fmt: str) -> None:
    assert format_for_name(raw) == fmt


def test_unsupported_format() -> None:
    with pytest.raises(Exception, match="unsupported file type"):
        format_for_name("book.mobi")


def test_epub_extraction(tmp_path: Path) -> None:
    path = _write_epub(
        tmp_path / "book.epub",
        ["Первый абзац.", "Второй абзац."],
    )
    document = extract_epub(path)

    assert document.format == "epub"
    assert document.page_count == 2
    assert "Первый абзац" in " ".join(page.text for page in document.pages)
    assert detect_format(path) == "epub"


def test_fb2_extraction_utf8_and_1251(tmp_path: Path) -> None:
    for encoding in ("utf-8", "cp1251"):
        path = _write_fb2(tmp_path / f"book-{encoding}.fb2", encoding)
        document = extract_fb2(path)
        assert document.format == "fb2"
        assert document.title == "FB2 Check"
        assert document.author == "Иван Петров"
        texts = " ".join(page.text for page in document.pages)
        assert "Первый абзац" in texts
        assert "Вложенный абзац" in texts
        labels = [page.printed_page_label for page in document.pages]
        assert "Глава 1 > Раздел 1.1" in labels
        assert detect_format(path) == "fb2"


def test_djvu_requires_tool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "book.djvu"
    path.write_bytes(b"AT&TFORM" + b"\x00" * 16)
    monkeypatch.setattr("shutil.which", lambda name: None)

    with pytest.raises(DjvuToolMissingError, match="djvutxt"):
        extract_djvu(path)


def test_djvu_pages_from_text_layer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "book.djvu"
    path.write_bytes(b"AT&TFORM" + b"\x00" * 16)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/djvutxt")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout="Первая страница\fВторая страница\f".encode(), stderr=b""
        ),
    )

    document = extract_djvu(path)

    assert document.format == "djvu"
    assert document.page_count == 2
    assert document.pages[1].text == "Вторая страница"


def test_djvu_without_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "book.djvu"
    path.write_bytes(b"AT&TFORM" + b"\x00" * 16)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/djvutxt")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout=b"\f", stderr=b""
        ),
    )

    with pytest.raises(DjvuNoTextError, match="no text layer"):
        extract_djvu(path)


def test_index_epub_sets_format(tmp_path: Path) -> None:
    path = _write_epub(tmp_path / "book.epub", ["Первый абзац.", "Второй абзац."])
    catalog = _catalog(tmp_path)
    client = QdrantClient(":memory:")
    indexer = _indexer(tmp_path, catalog, client)
    try:
        book = indexer.index_pdf(path)
        assert book.status is BookStatus.ready
        assert book.format == "epub"
        records, _ = client.scroll("books", limit=10, with_payload=True)
        assert {record.payload["book_format"] for record in records} == {"epub"}
    finally:
        indexer.close()
        client.close()


def test_index_fb2_sets_section_labels(tmp_path: Path) -> None:
    path = _write_fb2(tmp_path / "book.fb2")
    catalog = _catalog(tmp_path)
    client = QdrantClient(":memory:")
    indexer = _indexer(tmp_path, catalog, client)
    try:
        book = indexer.index_pdf(path)
        assert book.format == "fb2"
        assert book.chunk_count > 0
        records, _ = client.scroll("books", limit=20, with_payload=True)
        labels = {record.payload.get("printed_page_label") for record in records}
        assert "Глава 1" in labels
    finally:
        indexer.close()
        client.close()


def test_upload_accepts_epub(tmp_path: Path) -> None:
    source = _write_epub(tmp_path / "source.epub", ["Один абзац."])
    settings = _settings(tmp_path)
    service = LibraryService(settings, _catalog(tmp_path), FakeIndexer())
    try:
        entry, job, note = service.save_upload(
            "book.epub", io.BytesIO(source.read_bytes())
        )
    finally:
        service.close()

    assert job is None and note is None
    assert entry.format == "epub"
    assert (settings.books_dir / "book.epub").is_file()


def test_parse_error_message_has_no_text() -> None:
    assert issubclass(PdfParseError, RuntimeError)


def test_fb2_rejects_entity_declarations(tmp_path: Path) -> None:
    path = tmp_path / "bomb.fb2"
    path.write_bytes(
        b'<?xml version="1.0"?><!DOCTYPE FictionBook [<!ENTITY x "boom">]>'
        b'<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">'
        b"<body><section><p>&x;</p></section></body></FictionBook>"
    )

    with pytest.raises(PdfParseError, match="entity"):
        extract_fb2(path)


def test_lexical_hits_carry_book_format(tmp_path: Path) -> None:
    path = _write_epub(tmp_path / "book.epub", ["Квантование synthetic."])
    catalog = _catalog(tmp_path)
    client = QdrantClient(":memory:")
    indexer = _indexer(tmp_path, catalog, client)
    try:
        indexer.index_pdf(path)
        hits = catalog.search_lexical("Квантование")
        assert hits
        assert hits[0].book_format == "epub"
    finally:
        indexer.close()
        client.close()


def test_epub_prologue_section_label(tmp_path: Path) -> None:
    path = _write_epub(tmp_path / "prologue.epub", ["Текст пролога."], ["Пролог"])
    document = extract_epub(path)
    assert document.pages[0].printed_page_label == "Пролог"
    result = chunk_document(document, "book-prologue", chunk_size=400, chunk_overlap=40)
    assert result.chunks[0].section_path == "Пролог"
    assert any(section.section_path == "Пролог" for section in result.sections)


def test_fb2_unnamed_section_does_not_inherit_previous(tmp_path: Path) -> None:
    path = tmp_path / "unnamed.fb2"
    xml = (
        '<?xml version="1.0"?><FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">'
        "<body><section><title><p>Глава 1</p></title><p>Первый текст.</p></section>"
        "<section><p>Без заголовка.</p></section></body></FictionBook>"
    )
    path.write_bytes(xml.encode("utf-8"))
    document = extract_fb2(path)
    result = chunk_document(document, "book-unnamed", chunk_size=400, chunk_overlap=40)
    unnamed = [chunk for chunk in result.chunks if "Без заголовка" in chunk.text]
    assert unnamed and unnamed[0].section_path is None


def test_fb2_utf16_entity_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "utf16-bomb.fb2"
    xml = (
        '<?xml version="1.0" encoding="utf-16"?>'
        '<!DOCTYPE FictionBook [<!ENTITY x "boom">]>'
        '<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">'
        "<body><section><p>&x;</p></section></body></FictionBook>"
    )
    path.write_bytes(xml.encode("utf-16"))

    with pytest.raises(PdfParseError, match="entity"):
        extract_fb2(path)
