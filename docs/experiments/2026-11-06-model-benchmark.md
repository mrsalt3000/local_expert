# Сравнение локальных chat-моделей: RAG и Agent (T105, 2026-11-06)

## Что сравниваем

Retrieval от чат-модели не зависит: индекс, `bge-m3`, `top_k=20`,
`scope=expected` и candidate limit одинаковы. Поэтому page recall/MRR — один
baseline, а модели сравниваются по generation/agent-слою:

- citation precision/recall по золотым страницам;
- faithfulness и citation support (фиксированный судья);
- отказы, `zero_precision`, длина/формат ответа;
- latency p50/p95;
- для Agent — page/source recall, число запросов и раундов.

## Модели

| slug | модель | размер | статус |
| --- | --- | ---: | --- |
| `ministral-3-3b` | Ministral 3 3B Instruct | 3.0 GB | скачана |
| `gemma3-4b` | Gemma 3 4B IT | 3.4 GB | скачана |
| `phi4-mini` | Phi-4-mini-instruct | 2.5 GB | скачана |
| `qwen3-4b-q4` | Qwen3-4B-Instruct q4_K_M | 2.6 GB | скачана |
| `qwen3-4b-q8` | Qwen3-4B-Instruct q8_0 | 4.4 GB | скачана |
| `deepseek-flash-ref` | DeepSeek Flash, cloud | — | reference, 20 кейсов |

Qwen3-4B q5/q6 не проверяются: в библиотеке Ollama нет таких тегов, импорт
GGUF с HF не делался. Текущая `qwen3.5:4b` не кандидат: она — фиксированный
судья для всех моделей.

## Методика

- RAG: `book-expert stage-eval data/eval/combined-120.local.json --limit 120
  --scope expected --faithfulness --judge-provider ollama --judge-model
  qwen3.5:4b`.
- Agent: `book-expert agent-eval ... --limit 20` с теми же флагами судьи.
- `--judge-model` добавлен специально: без него `--judge-provider ollama`
  оставлял судьёй саму кандидатскую модель.
- `BOOK_EXPERT_OLLAMA_THINK=false`, `BOOK_EXPERT_TOP_K=20`, температура
  провайдера по умолчанию.
- Отчёты: `data/artifacts/model-benchmark/<slug>-{stage,agent}.json`,
  сводка — `summary.json` и `summary.md`.

## Smoke (1 кейс, не результат)

| модель | answer p50 | отказ | faithfulness |
| --- | ---: | --- | ---: |
| Ministral 3 3B | 17.8 s | да | — |
| Gemma 3 4B | 18.4 s | нет | 1.0 |
| Phi-4-mini | 21.8 s | нет | 1.0 |
| Qwen3-4B q4 | 27.2 s | нет | 1.0 |
| Qwen3-4B q8 | **59.3 s** | нет | 1.0 |

q8 в 2.2 раза медленнее q4 уже на одном кейсе — вероятно, часть слоёв уходит
на CPU при 6 ГБ VRAM. Это главный риск квантования; итоговый вывод — после
полного прогона.

## Статус

Полный прогон 120 RAG + 20 Agent запущен в фоне 2026-11-06; на момент отчёта
идёт первая модель (~20/120). Оценка темпа: ~90 минут на модель для stage;
q8 ожидаемо дольше. Итоговые числа появятся в `summary.md` после завершения
всех моделей; до этого smoke-таблицу нельзя использовать как ranking.

## Ограничения

- Судья `qwen3.5:4b` имеет human-holdout kappa `.456` (T103): метрики
  faithfulness/support — только relative ranking, не абсолютное качество.
- Один прогон на модель, без повторов и температуры; разброс не измерен.
- DeepSeek Flash — только reference на 20 кейсах, не кандидат в прод.
- q5/q6 Qwen3 не проверены; вывод о квантовании ограничен q4 vs q8.
