# Parser Assessment: Contract и Rubric

## Цель

Сравнить два PDF parser-а — **Docling** и **PyMuPDF** — на фиксированном
local-only sample реальной книги. Каждая страница оценивается reviewer-ом по
критериям ниже и фиксируется в page-level assessment record.

## Границы данных

- Реальный PDF книги — private input: не копируется, не индексируется и не
  попадает в Git.
- Parser outputs (extracted text, OCR text, структуры) и реальный manifest
  (`data/eval/parser_assessment.local.json`) **не коммитятся**.
- В Git разрешены только schema, rubric, synthetic fixtures и этот контракт.
- В tracked файлах запрещены: реальные номера страниц, названия книг, авторы,
  цитаты, source/extracted/OCR text, passage/chunk text и абсолютные пути.

## Artifact hierarchy

- **Manifest** (`parser-assessment-manifest.schema.json`): один parser run по
  local-only sample; envelope с массивом page-level records (`records[]`).
  Manifest schema ссылается на page-level schema через локальный `$ref`.
- **Page-level record** (`parser-assessment.schema.json`): оценка одной
  страницы одним parser configuration.
- **Aggregate report**: будущий артефакт (сводка по всему sample) — не входит
  в текущий contract.

## Page identifiers

У каждой страницы два независимых идентификатора:

| Поле | Смысл |
| --- | --- |
| `pdf_page_index` | 0-based индекс страницы внутри PDF-файла (порядок в PDF). |
| `printed_page_label` | Метка, напечатанная на физической странице книги. |

Оба обязательны: PDF-индекс нужен для машинной навигации, printed label — для
сверки с реальной книгой. Реальные значения живут только в local-only manifest.

## Статусы

- `pass` — критерий выполнен полностью.
- `warning` — выполнено с оговорками, требующими внимания reviewer-а.
- `fail` — критерий нарушен.
- `not_applicable` — критерий не применим к данному типу страницы.

## Критерии (9)

1. `text_coverage` — полнота захвата текста страницы.
2. `reading_order` — порядок чтения соответствует логике документа.
3. `heading_structure` — корректность иерархии заголовков.
4. `table_fidelity` — точность воспроизведения таблиц.
5. `footnote_separation` — сноски отделены от основного текста.
6. `header_footer_handling` — колонтитулы корректно обработаны.
7. `image_handling` — обработка изображений и диаграмм.
8. `page_linkage` — корректная привязка контента к странице.
9. `runtime` — производительность (время обработки страницы).

## Типы страниц (anonymized categories)

Категории из schema, без реальных данных книги — только тип контента:
`body_text`, `mixed_language`, `table_of_contents`, `table`, `footnote`,
`header_footer`, `image_or_diagram`, `callout`, `heading_hierarchy`,
`end_boundary`.

## Acceptance rule

`fail` по **text_coverage**, **reading_order** или **page_linkage** на странице
типа `body_text` блокирует принятие parser-а как единственного baseline path —
до investigation и повторной оценки.

## Baseline run

- OCR **выключен** в первом native-text baseline run: оценивается только
  встроенный текстовый слой PDF.
- Table structure extraction — отдельный измеряемый параметр, не часть
  `text_coverage`.

## Docling vs PyMuPDF (2026-11-01)

Docling подключён как optional extra (`uv sync --extra docling`), адаптер —
`ingestion/docling.py`, сравнение — `scripts/parser_compare.py`.

На локальном sample из 5 страниц реальной книги; оба парсера таймились
только на этих страницах:

| Метрика | PyMuPDF | Docling |
| --- | ---: | ---: |
| nonempty coverage | 1.0 | 1.0 |
| chars | 17 992 | 10 675 |
| blocks | 87 | 30 |
| elapsed, ms | ~218 | ~6 154 |

`mean_token_jaccard` = **0.854**, `mean_token_recall` = **0.846** (multiset,
без потери повторов). Две страницы дали token recall **0.63** и **0.73**:
Docling классифицировал часть body text как `PictureItem`, а при
`do_ocr=false` этот текст теряется; `force_backend_text=true` этого не
исправляет. Картинки теперь не добавляют placeholder в текст.

Вывод: **PyMuPDF остаётся baseline для текста**. Docling можно использовать
только как дополнительный structural parser с OCR/table extraction и ручной
проверкой; заменять им native-text path нельзя.

Реальные page indices есть только в git-ignored локальном отчёте
`data/artifacts/parser-compare.local.json`; текст книги и пути в него не
пишутся.

## Ссылки

- Manifest schema: `../schemas/parser-assessment-manifest.schema.json`
- Page-level schema: `../schemas/parser-assessment.schema.json`
- Synthetic fixtures:
  `../tests/fixtures/parser_assessment_manifest.synthetic.json`,
  `../tests/fixtures/parser_assessment.synthetic.json`
- Local-only policy: `../data/eval/README.md`
