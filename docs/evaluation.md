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
| `page_recall@k` / `page_mrr@k` | То же по страницам: устойчиво к переиндексации и к смене chunk id; страница сравнивается парой `(book_id, pdf_page_index)`, совпадение номера в другой книге не засчитывается |
| `page_recall@5` / `chunk_recall@5` | Метрики на первых пяти хитах; не зависят от `--k`, поэтому сравнимы между прогонами с разным k |
| `hit@k` / `page_hit@k` (multi-gold) | Найден ли **хотя бы один** допустимый фрагмент/страница. Основная метрика, когда у вопроса несколько равноправных источников |
| `MAP@k` / `nDCG@k` (multi-gold) | Ранжирование по нескольким допустимым фрагментам; nDCG учитывает grade 2 (gold) и 1 (та же страница/раздел/дубль). Справочные: падают, когда допустимых фрагментов много, и «найти все» не требуется |
| `faithfulness` | Доля утверждений ответа, поддержанных хотя бы одним найденным фрагментом (LLM-судья, `--faithfulness`) |
| `source_page_recall` | Доля ожидаемых страниц, попавших в найденные фрагменты (для «Агента» — до отбора цитат) |
| `citation_support_precision` | Доля утверждений с маркером `[n]`, где фрагмент `n` действительно поддерживает утверждение |
| `citation_usage` | Доля показанных цитат, которые вообще упомянуты маркером хотя бы в одном утверждении |
| `citation_support_rate` | Доля использованных цитат, которые поддерживают хотя бы одно утверждение. Не зависит от длины списка цитат, в отличие от page-overlap precision |
| `citation_coverage` | Доля утверждений, у которых вообще есть маркер цитаты |
| `claims_truncated` | Сколько ответов судья обрезал по `MAX_CLAIMS` (48): длиннее — метрика покрывает не весь ответ |

Длинные ответы судья оценивает батчами по `JUDGE_BATCH_SIZE` (12) утверждений:
на 24–38 утверждениях одна большая JSON-просьба к локальной 4B-модели ломала
формат (16 из 20 ответов «Агента»), батчи это устраняют.

**Две шкалы citation-метрик.** `citation_precision` в `stage-eval` — это
page-overlap: сколько цитируемых страниц совпало с размеченной. Она дешёвая,
но занижает качество, когда верный ответ опирается на соседние страницы или на
другую книгу с тем же фактом. `citation_support_precision` из
`--faithfulness` проверяет именно поддержку утверждения фрагментом (как
citation precision в главе 6 книги). Для решений о качестве смотреть на вторую,
page-overlap держать как sanity-check.

**Citation gauge (T85).** Page-overlap precision не может превысить `1/N`,
поэтому при пяти цитатах её потолок — `.20`. `citation_usage` и
`citation_support_rate` считают по каждой цитате: используется ли она в
утверждении и поддерживает ли его. Живой прогон «Агента» (20 кейсов, cap 5):
page-overlap precision **.1775**, но `citation_usage` **1.0** (все пять цитат
упомянуты) и `citation_support_rate` **.9368**; у 4 ответов с нулевой
page-overlap precision поддержка ≥ .8. То есть низкая precision — арифметика
`1/5`, а не плохие цитаты. Детали:
[experiments/2026-11-04-citation-gauge.md](experiments/2026-11-04-citation-gauge.md).

**Multi-gold (T83).** Single-gold `chunk_recall` засчитывает только точное
совпадение `chunk_id`, поэтому формально занижен, если верный фрагмент лежит
рядом. Проверка на 69 и 120 кейсах показала, что на практике альтернатив
почти нет: при требовании текстового совпадения с gold (Jaccard 3-грамм
≥ 0.3) нашлось 3 фрагмента на 69 кейсов и 0 на 120, а chunk hit@k вырос на
один кейс. Наивное правило «любой чанк той же страницы» отвергнуто: в 24/24
изменившихся кейсах такой «допустимый» чанк текстовно не связан с gold
(Jaccard 0.00) и просто дублирует page recall. `stage-eval --multigold <file>`
и `agent-eval --multigold <file>` добавляют `retrieval_multigold` (hit,
recall, MAP@k, nDCG@k), не меняя single-gold дефолт. Разметку строит
`scripts/multigold_eval.py`. Детали:
[experiments/2026-11-03-multigold.md](experiments/2026-11-03-multigold.md).

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
`--judge-provider ollama|openai|deepseek` выносит судью на другую модель,
чем ответы (кросс-модельная проверка); выбранный судья пишется в `meta`
как `judge_provider`/`judge_chat_model`.

`--strict-judge` требует явного предложения в источнике: частичное совпадение
темы не считается поддержкой. На 32 парах строгий промпт снизил принятие
частично поддержанных утверждений с **46.9%** до **28.1%**, потеряв на
верных 93.8% → **90.6%**. Флаг opt-in, по умолчанию выключен; в `meta`
пишется `judge_strict`.

`--quote-judge` требует дословную цитату: судья возвращает `quote`, и
поддержка засчитывается только если она найдена подстрокой в цитируемом
фрагменте (нормализация пробелов/ё). На 32 парах валидная цитата была у
**30/32** верных и **16/32** частичных. Это проверяемое доказательство,
поэтому режим пригоден для аудита и гейтов; в `meta` пишется
`judge_mode="quote"`.

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
