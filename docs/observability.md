# Наблюдаемость и SLO (T90, 2026-11-04)

## Что пишется

`RequestMetrics` (`src/book_expert/web/metrics.py`) ведёт счётчики запросов и
дописывает каждое событие в `data/artifacts/metrics.local.jsonl`:

```json
{"recorded_at": "2026-11-04T17:00:00+00:00", "mode": "rag",
 "elapsed_ms": 1712.3, "refused": false, "error": false}
```

- Полный текст вопроса/ответа и цитаты **не пишутся**.
- При старте сервис читает последние записи, поэтому `requests/refusals/errors`
  и окно latency переживают рестарт.
- Окно latency — последние 200 запросов; в `by_mode` — отдельное окно на режим.
- Битые строки JSONL пропускаются; ошибка записи на диск не ломает запрос.

## SLO

| Режим | p50 | p95 | max refusal rate |
|---|---:|---:|---:|
| `rag` | 2.5 s | 5 s | 15% |
| `advanced` | 4 s | 8 s | 15% |
| `super` | 8 s | 15 s | 20% |
| `agent` | 8 s | 15 s | 20% |

`GET /api/health` отдаёт `by_mode` и `slo`:

```json
{
  "requests": 42,
  "refusals": 3,
  "by_mode": {
    "rag": {
      "requests": 40,
      "refusals": 3,
      "refusal_rate": 0.075,
      "latency_ms": {"p50": 1712, "p95": 2400},
      "slo": {"p50_ok": true, "p95_ok": true, "refusal_ok": true, "all_ok": true}
    }
  },
  "slo": {"persisted": true, "targets": {"rag": {"p50_ms": 2500, ...}}}
}
```

## Ограничения

- **Token/cost не учитываются.** `ChatResult.usage` есть у провайдеров, но
  наружу через `Answer`/`ChatResponse` не пробрасывается; нужен отдельный
  проход по провайдерам и режимам.
- **Корреляция feedback ↔ качество не реализована.** `feedback.local.jsonl`
  хранит rating/mode/chat_id, а метрики — только latency/refusals; для
  корреляции нужен общий ключ запроса и прогон eval.
- SLO-пороги захардкожены в `SLO`; для изменения нужен код/тест, env-ручки нет.
- `uptime_s` считается от текущего процесса, а `requests` — от всего файла;
  это ожидаемое поведение, но не «uptime с первого запроса».
