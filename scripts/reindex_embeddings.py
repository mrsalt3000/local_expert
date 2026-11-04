"""Resumable full-catalog reindex into a separate Qdrant collection.

Reads chunks from the SQLite catalog, embeds them with the given Ollama model
and upserts into ``--collection``. Books that already have all their points in
the target collection are skipped, so the command can be restarted safely.
"""

from __future__ import annotations

import argparse
import sqlite3
import time
import uuid
from pathlib import Path

from qdrant_client import QdrantClient, models

from book_expert.config import Settings
from book_expert.ingestion.indexer import PAYLOAD_INDEXES
from book_expert.providers import get_provider


def _upsert_with_retries(
    client: QdrantClient,
    collection: str,
    points: list[models.PointStruct],
    *,
    attempts: int = 4,
) -> None:
    """Upsert с повтором: Qdrant может отдать таймаут на оптимизации сегментов."""
    for attempt in range(1, attempts + 1):
        try:
            client.upsert(collection_name=collection, points=points, wait=True)
            return
        except Exception as exc:
            if attempt == attempts:
                raise
            delay = 2**attempt
            print(
                f"upsert retry {attempt}/{attempts - 1} after {exc!r}; "
                f"sleeping {delay}s",
                flush=True,
            )
            time.sleep(delay)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3-embedding:0.6b")
    parser.add_argument("--collection", default="books_qwen3emb_full")
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument(
        "--catalog", type=Path, default=Path("data/artifacts/catalog.sqlite3")
    )
    args = parser.parse_args()

    settings = Settings(
        provider="ollama",
        embedding_model=args.model,
        embedding_dim=args.dim,
    )
    provider = get_provider(settings)
    api_key = (
        settings.qdrant_api_key.get_secret_value()
        if settings.qdrant_api_key is not None
        else None
    )
    client = QdrantClient(url=settings.qdrant_url, api_key=api_key, timeout=120)
    db = sqlite3.connect(args.catalog)

    if not client.collection_exists(args.collection):
        client.create_collection(
            collection_name=args.collection,
            vectors_config=models.VectorParams(
                size=args.dim, distance=models.Distance.COSINE
            ),
        )
        for field_name, schema in PAYLOAD_INDEXES:
            client.create_payload_index(
                collection_name=args.collection,
                field_name=field_name,
                field_schema=schema,
                wait=True,
            )

    books = db.execute(
        "select id, title, format, chunk_count from books "
        "where status='ready' and chunk_count>0 order by chunk_count desc"
    ).fetchall()
    total_chunks = sum(row[3] for row in books)
    started = time.perf_counter()
    done_chunks = 0
    skipped_chunks = 0
    try:
        for position, (book_id, title, book_format, chunk_count) in enumerate(books, 1):
            book_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="book_id", match=models.MatchValue(value=book_id)
                    )
                ]
            )
            count = client.count(
                collection_name=args.collection,
                count_filter=book_filter,
                exact=True,
            ).count
            if count >= chunk_count:
                done_chunks += chunk_count
                skipped_chunks += chunk_count
                continue
            client.delete(
                collection_name=args.collection,
                points_selector=models.FilterSelector(filter=book_filter),
                wait=True,
            )
            chunks = db.execute(
                "select id, text, pdf_page_index, printed_page_label, section_path, "
                "ordinal from chunks where book_id=? order by ordinal",
                (book_id,),
            ).fetchall()
            for start in range(0, len(chunks), args.batch):
                batch = chunks[start : start + args.batch]
                texts = []
                for _, text, _, _, section, _ in batch:
                    parts = [f"Книга: {title}"]
                    if section:
                        parts.append(f"Раздел: {section}")
                    parts.append(text)
                    texts.append("\n".join(parts))
                vectors = provider.embed(texts).vectors
                points = [
                    models.PointStruct(
                        id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id)),
                        vector=list(vector),
                        payload={
                            "book_id": book_id,
                            "book_title": title,
                            "book_format": book_format,
                            "chunk_id": chunk_id,
                            "text": text,
                            "pdf_page_index": page,
                            "printed_page_label": label,
                            "section_path": section,
                            "ordinal": ordinal,
                        },
                    )
                    for (chunk_id, text, page, label, section, ordinal), vector in zip(
                        batch, vectors, strict=True
                    )
                ]
                _upsert_with_retries(client, args.collection, points)
                done_chunks += len(points)
            elapsed = time.perf_counter() - started
            embedded = done_chunks - skipped_chunks
            rate = embedded / elapsed if elapsed else 0
            remaining = (total_chunks - done_chunks) / rate if rate else 0
            print(
                f"{position}/{len(books)} {title[:35]}: "
                f"{done_chunks}/{total_chunks} chunks, {rate:.1f}/s, "
                f"eta {remaining / 3600:.2f}h",
                flush=True,
            )
    finally:
        client.close()
        close = getattr(provider, "close", None)
        if callable(close):
            close()
    print(f"done: {done_chunks} chunks in {args.collection}")


if __name__ == "__main__":
    main()
