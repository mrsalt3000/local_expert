# Evaluation Data

Здесь разрешены только **небольшие вручную созданные metadata/evaluation
fixtures** без защищённого текста книги.

Запрещено класть: реальный PDF, извлечённый текст, embeddings, индексы, логи
запросов и любые приватные артефакты.

## Parser assessment manifests

- `parser_assessment.local.example.json` — **synthetic manifest example**:
  shape-only template для локального manifest, разрешён в Git.
- `parser_assessment.local.json` — **local-only real manifest** (git-ignored):
  реальные printed page numbers и private reviewer notes; не коммитится.
- Manifest schema и page-level schema — **разные contracts**: manifest
  (`schemas/parser-assessment-manifest.schema.json`) — envelope с `records[]`
  для batch/local run; page-level (`schemas/parser-assessment.schema.json`) —
  оценка одной страницы одним parser configuration.
- В Git разрешены только:
  - `.gitignore`
  - `README.md`
  - `parser_assessment.local.example.json`
- Content-bearing поля (source/extracted/OCR text, цитаты, passage/chunk text,
  пути к файлам) и реальные данные книги в tracked файлах не допускаются.

Ссылки: [manifest schema](../../schemas/parser-assessment-manifest.schema.json),
[page-level schema](../../schemas/parser-assessment.schema.json).

Формат и rubric: см. [docs/parsing-assessment.md](../../docs/parsing-assessment.md).
