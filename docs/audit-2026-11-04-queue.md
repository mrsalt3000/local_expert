# Очередь T86–T92: итоги на 2026-11-04

Аудит 2026-11-04 дал **7.9/10**; после него выполнены все задачи очереди:
T86, T87, T88, T89, T90, T91 и T92. Незакрытыми остаются только
ограничения внутри T90: token/cost и корреляция feedback ↔ качество.

## Принятые задачи

| Задача | Коммит | Результат |
| --- | --- | --- |
| T86 — калибровка судьи | `add1b4b` | 152 утверждения, 19 ответов; DeepSeek binary acc .855 / kappa .715, 4B binary acc .839 / kappa .679, quote kappa .739 на 24 утверждениях. Разметка независимым агентом, не человеком. |
| T87 — ответы без маркеров | `2f18477` | Убрана подстановка всех хитов; 69 кейсов: отказы 6→8, citation precision .692→.696, recall .7826→.7681; runtime-проверка opt-in, p50 1.84→2.63 s (1.43×). |
| T88 — каталог-wide eval | `9704b03` | 120 кейсов / 30 книг: routing top1 .667, page recall@5 .775, отказы 17/120; Qdrant timeout 5→30 s. |
| T89 — ingestion hygiene | `8e3d6f3` | Reconciliation перевёл 1 зависшую `indexing` в `failed`; каталог 1160 ready / 0 indexing / 23 failed; byte-aware batching 8 MiB. Живой retry 465-стр. книги упёрся в замаскированную `qdrant unavailable`. |
| T90 — метрики и SLO | `23942c3` | JSONL-журнал переживает рестарт; `/api/health` отдаёт `by_mode` и SLO; 498 тестов. |
| T91 — герметичность тестов | `969398a` | Autouse-fixture чистит `BOOK_EXPERT_*`; 494 теста зелёные при 33 экспортированных переменных. |

## Проверки

`pytest -q` — 498 passed; `ruff format --check`, `ruff check`, `mypy src`,
`pre-commit --all-files` — зелёные. CI на `23942c3` — success
(`41856499640`). Ветка `main` синхронизирована с `origin/main`.
