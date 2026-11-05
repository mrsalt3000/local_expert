# Наблюдаемость, стоимость и feedback (T90 + T104, 2026-11-06)

## Что пишется

`RequestMetrics` (`src/book_expert/web/metrics.py`) ведёт счётчики запросов и
дописывает каждое событие в `data/artifacts/metrics.local.jsonl`:

```json
{"recorded_at": "2026-11-06T10:00:00+00:00", "mode": "rag",
 "elapsed_ms": 1712.3, "refused": false, "error": false,
 "provider": "deepseek", "model": "deepseek-chat", "request_id": "…",
 "citations": 5, "unsupported_claims": 1,
 "tokens": {"prompt_tokens": 1200, "completion_tokens": 300, "total_tokens": 1500},
 "cost_usd": 0.0012}
```

- Полный текст вопроса/ответа и цитаты **не пишутся**: только режим,
  задержка, флаги, число токенов/цитат и стоимость.
- `request_id` — общий ключ с `feedback.local.jsonl`; его возвращает
  `POST /api/chat` и SSE-событие `done`.
- `tokens` пишутся для обоих путей: обычный `/api/chat` и SSE
  `/api/chat/stream`. Для `advanced/super` с жёстким таймаутом usage
  возвращается из дочернего процесса внутри `Answer`.
- Usage собирается обёрткой `MeteredProvider` вокруг провайдера в пределах
  одного запроса; агент складывает `usage_metadata` из LangChain-сообщений.
- При старте сервис читает последние записи, поэтому `requests/refusals/errors`,
  окно latency и токены переживают рестарт.
- Окно latency — последние 200 запросов; в `by_mode` — отдельное окно на режим.
- Битые строки JSONL пропускаются; ошибка записи на диск не ломает запрос.

## Стоимость

Ставки задаются env-переменными (за 1M токенов, в валюте провайдера):

```bash
BOOK_EXPERT_PRICE_INPUT_PER_MTOK=0.27
BOOK_EXPERT_PRICE_OUTPUT_PER_MTOK=1.10
```

По умолчанию обе ставки `0.0`: токены считаются всегда, `cost_usd` остаётся
нулевым, пока ставки не заданы. Секреты и API-ключи в метрики не попадают.

## SLO

| Режим | p50 | p95 | max refusal rate |
|---|---:|---:|---:|
| `rag` | 2.5 s | 5 s | 15% |
| `advanced` | 4 s | 8 s | 15% |
| `super` | 8 s | 15 s | 20% |
| `agent` | 8 s | 15 s | 20% |

`GET /api/health` отдаёт `by_mode`, `by_provider`, `tokens`, `cost_usd`,
`usage_coverage` и `slo`:

```json
{
  "requests": 42,
  "tokens": {"prompt": 50000, "completion": 8000, "total": 58000},
  "cost_usd": 0.0123,
  "usage_coverage": 0.95,
  "by_mode": {
    "rag": {
      "requests": 40,
      "refusals": 3,
      "refusal_rate": 0.075,
      "latency_ms": {"p50": 1712, "p95": 2400},
      "tokens": {"prompt_tokens": 48000, "completion_tokens": 7600,
                 "total_tokens": 55600},
      "cost_usd": 0.0118,
      "slo": {"p50_ok": true, "p95_ok": true, "refusal_ok": true, "all_ok": true}
    }
  },
  "by_provider": {"deepseek": {"requests": 42, "total_tokens": 58000,
                                "cost_usd": 0.0123}},
  "slo": {"persisted": true, "targets": {"rag": {"p50_ms": 2500, ...}}}
}
```

`usage_coverage` — доля запросов, для которых провайдер вернул usage. Если
провайдер не отдаёт токены (например, часть Ollama-ответов), coverage ниже 1.

## Feedback correlation

`scripts/feedback_correlation.py` соединяет `metrics.local.jsonl` и
`feedback.local.jsonl` по `request_id` и считает связь оценки пользователя с
долей неподдержанных утверждений:

```bash
uv run python scripts/feedback_correlation.py
```

Выход — `data/artifacts/feedback-correlation.json`: число пар, средняя
`unsupported_rate` для `up`/`down`, point-biserial корреляция и сами пары без
текста ответов. Это не заменяет offline faithfulness: runtime знает только
число claim'ов, которые не прошли citation verification, а не полный разбор.

## Ограничения

- Sliding window latency — 200 запросов; token/cost считаются по всему файлу,
  как `requests`.
- Стоимость считается по одной паре ставок на активного провайдера; разные
  модели одного провайдера требуют отдельных ставок.
- `unsupported_claims` заполняется только при включённой проверке цитат;
  иначе корреляция идёт по нулям.
- Токены embedding-вызовов не учитываются: провайдеры не возвращают для них
  usage, поэтому в cost входит только chat.
- SLO-пороги захардкожены в `SLO`; для изменения нужен код/тест, env-ручки нет.
- `uptime_s` считается от текущего процесса, а `requests` — от всего файла;
  это ожидаемое поведение, но не «uptime с первого запроса».
