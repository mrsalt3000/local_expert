# Очередь T86–T92: итоги на 2026-11-04

Аудит 2026-11-04 дал **7.9/10**; после него выполнены T86, T87, T88, T89,
T91 и эта запись T92. T90 (наблюдаемость/стоимость) остаётся открытой: нужны
персистентные метрики, token/cost и SLO.

## Принятые задачи

| Задача | Коммит | Результат |
| --- | --- | --- |
| T86 — калибровка судьи | `add1b4b` | 152 утверждения, 19 ответов; DeepSeek binary acc .855 / kappa .715, 4B binary acc .839 / kappa .679, quote kappa .739 на 24 утверждениях. Разметка независимым агентом, не человеком. |
| T87 — ответы без маркеров | `2f18477` | Убрана подстановка всех хитов; 69 кейсов: отказы 6→8, citation precision .692→.696, recall .7826→.7681; runtime-проверка opt-in, p50 1.84→2.63 s (1.43×). |
| T88 — каталог-wide eval | `9704b03` | 120 кейсов / 30 книг: routing top1 .667, page recall@5 .775, отказы 17/120; Qdrant timeout 5→30 s. |
| T89 — ingestion hygiene | `8e3d6f3` | Reconciliation перевёл 1 зависшую `indexing` в `failed`; каталог 1160 ready / 0 indexing / 23 failed; byte-aware batching 8 MiB. Живой retry 465-стр. книги упёрся в замаскированную `qdrant unavailable`. |
| T91 — герметичность тестов | `969398a` | Autouse-fixture чистит `BOOK_EXPERT_*`; 494 теста зелёные при 33 экспортированных переменных. |

## Открытые задачи

- **T90** — персистентные метрики, token/cost, SLO, корреляция feedback.
  Не начат: требует изменения `web/metrics.py`, API и документации SLO.

## Проверки

`pytest -q` — 494 passed; `ruff format --check`, `ruff check`, `mypy src`,
`pre-commit --all-files` — зелёные. CI на `9704b03` — success
(`41852177512`). Ветка `main` синхронизирована с `origin/main`.
