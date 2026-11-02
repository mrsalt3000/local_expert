# Evaluation

Оценка качества retrieval и grounded answer на synthetic-наборе без реального
текста книги. Реальные прогоны используют local-only данные и не коммитятся.

## Fixtures

Разрешены только небольшие вручную созданные metadata/evaluation fixtures без
защищённого текста книги. См. [data/eval/README.md](../data/eval/README.md).
Пример набора: `tests/fixtures/eval_cases.synthetic.json`.

## Метрики

| Метрика | Что измеряет |
| --- | --- |
| `recall@k` | Доля ожидаемых chunk id, попавших в top-k |
| `MRR` | 1 / rank первого релевантного chunk, усреднённый по вопросам |
| `citation_precision` | Доля cited chunk id, которые ожидались |
| `citation_recall` | Доля ожидаемых chunk id, попавших в citations |
| `faithfulness` | Доля утверждений ответа, поддержанных хотя бы одним найденным фрагментом (LLM-судья, `--faithfulness`) |

## Процедура

1. Для быстрой проверки метрик без реальных данных:

```bash
uv run book-expert eval --synthetic --json
```

`--synthetic` засеивает вымышленный корпус в каталог, прогоняет lexical-only
eval и удаляет его после завершения.

2. Для реального набора подготовить JSON: `question`, `expected_chunk_ids`,
   `expected_book_ids` (опционально).
3. Убедиться, что книги проиндексированы и Qdrant/Ollama доступны.
4. Запустить:

```bash
uv run book-expert eval path/to/cases.json --k 5 --json
```

Без Qdrant используйте `--lexical-only`; для авто-routing — `--route`.

5. Сохранить отчёт в `data/artifacts/` (git-ignored) или приложить сводку без
   текста книги.
6. Для воспроизводимости фиксировать модель, embedding model, версию Qdrant и
   commit в описании прогона.

## Пороги приёмки (v1)

- `recall@5 >= 0.8` на synthetic-наборе.
- `MRR >= 0.6`.
- `citation_precision >= 0.7`.
- Нет случаев `fail` по `text_coverage` на страницах `body_text` в parser
  assessment.

Пороги пересматриваются после первого прогона на реальной библиотеке; реальные
значения остаются локальными.

## Parser assessment

Отдельный контракт и rubric — [docs/parsing-assessment.md](parsing-assessment.md).
Автоматизированный PyMuPDF-прогон пишет `data/eval/parser_assessment.local.json`
и требует ручного review по 9 критериям.
