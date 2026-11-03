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
| `page_recall@k` / `page_mrr@k` | То же по страницам: устойчиво к переиндексации и к смене chunk id |
| `page_recall@5` / `chunk_recall@5` | Метрики на первых пяти хитах; не зависят от `--k`, поэтому сравнимы между прогонами с разным k |
| `faithfulness` | Доля утверждений ответа, поддержанных хотя бы одним найденным фрагментом (LLM-судья, `--faithfulness`) |
| `source_page_recall` | Доля ожидаемых страниц, попавших в найденные фрагменты (для «Агента» — до отбора цитат) |
| `citation_support_precision` | Доля утверждений с маркером `[n]`, где фрагмент `n` действительно поддерживает утверждение |
| `citation_coverage` | Доля утверждений, у которых вообще есть маркер цитаты |

**Две шкалы citation-метрик.** `citation_precision` в `stage-eval` — это
page-overlap: сколько цитируемых страниц совпало с размеченной. Она дешёвая,
но занижает качество, когда верный ответ опирается на соседние страницы или на
другую книгу с тем же фактом. `citation_support_precision` из
`--faithfulness` проверяет именно поддержку утверждения фрагментом (как
citation precision в главе 6 книги). Для решений о качестве смотреть на вторую,
page-overlap держать как sanity-check.

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
uv run book-expert stage-eval path/to/cases.json --scope candidates --relevance-filter
uv run book-expert stage-eval path/to/cases.json --scope candidates --k 10 --faithfulness
uv run book-expert agent-eval path/to/cases.json --limit 20 --faithfulness
```

`agent-eval` прогоняет `SuperAdvancedAnswerer` — путь, который UI открывает
по умолчанию; метрики те же, page recall считается по цитатам ответа.
Без Qdrant используйте `--lexical-only`; для авто-routing — `--route`.
`stage-eval --k` задаёт число хитов в ответе (по умолчанию
`BOOK_EXPERT_TOP_K`), `--faithfulness` включает LLM-судью утверждений.

5. Сохранить отчёт в `data/artifacts/` (git-ignored) или приложить сводку без
   текста книги.
6. Воспроизводимость обеспечивает сам отчёт: `stage-eval` пишет блок `meta` с
   провайдером, chat/embedding-моделью, `top_k`, `k`, scope, флагом
   relevance-фильтра, коллекцией Qdrant, commit, sha256 датасета, временем и
   списком экспортированных `BOOK_EXPERT_*`. `agent-eval` добавляет
   `advanced_query_limit`, `super_scope_limit`, `super_refine_enabled` и
   `answer_max_citations`. Переменные окружения имеют
   приоритет над `.env`, поэтому при сравнении прогонов сверяйте `meta`.

## Пороги приёмки (v1)

- `recall@5 >= 0.8` на synthetic-наборе.
- `MRR >= 0.6`.
- `citation_precision >= 0.7`.
- Нет случаев `fail` по `text_coverage` на страницах `body_text` в parser
  assessment.

Пороги пересматриваются после первого прогона на реальной библиотеке; реальные
значения остаются локальными.

**Chunk- и page-шкалы.** Порог `recall@5 >= 0.8` относится к страничным
метрикам (`page_recall@5`), потому что chunk id привязан к конкретной сборке
индекса и расходится после переиндексации. Chunk-метрики остаются в отчёте
справочно: на 120 кейсах chunk recall@5 ≈ 0.62 при page recall@5 ≈ 0.81–0.83,
и низкий chunk recall сам по себе не означает провал retrieval. Для citation
качества основной считается `citation_support_precision` (поддержка
утверждения), page-overlap — sanity-check.

## Parser assessment

Отдельный контракт и rubric — [docs/parsing-assessment.md](parsing-assessment.md).
Автоматизированный PyMuPDF-прогон пишет `data/eval/parser_assessment.local.json`
и требует ручного review по 9 критериям.

## Правило решений о качестве

- **Retrieval, routing и citations** сравнивать только на наборе
  `multi-book-40` (69 reviewed-кейсов, 10 книг). Одиночный срез из 20 кейсов
  шумит по recall (.60–.80 на одних и тех же дефолтах) и годится лишь для
  быстрой проверки гипотезы.
- **Generation-метрики** (faithfulness, claim-level support) устойчивы и на
  20 кейсах, но итоговые числа фиксировать на полном наборе.
- Любое изменение, влияющее на качество, подтверждать A/B на 69 кейсах или
  на 120 (5 книг) + 69 (10 книг), а не на одном малом срезе.
