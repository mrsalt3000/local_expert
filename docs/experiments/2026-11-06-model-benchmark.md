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
| `ministral-3-3b` | Ministral 3 3B Instruct | 3.0 GB | 120 + 20 |
| `gemma3-4b` | Gemma 3 4B IT | 3.4 GB | 120 + 20 |
| `phi4-mini` | Phi-4-mini-instruct | 2.5 GB | 120 + 20 |
| `qwen3-4b-q4` | Qwen3-4B-Instruct q4_K_M | 2.6 GB | 120 + 20 |
| `qwen3-4b-q8` | Qwen3-4B-Instruct q8_0 | 5.7 GB | **failed: timeout** |
| `deepseek-flash-ref` | DeepSeek Flash, cloud | — | 20 RAG, reference |

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

## RAG, 120 кейсов

| модель | n | cit prec | cit rec | faith | support | отказы | p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ministral 3 3B | 120 | .292 | .408 | .906 | .774 | **57** | 17.1 s |
| Gemma 3 4B | 120 | .373 | .717 | .901 | .788 | 7 | 18.3 s |
| Phi-4-mini | 120 | .321 | .667 | .870 | .728 | 26 | **11.1 s** |
| **Qwen3-4B q4** | 120 | **.530** | **.817** | **.939** | **.868** | **2** | 13.6 s |
| Qwen3-4B q8 | — | — | — | — | — | — | timeout |
| DeepSeek Flash ref | 20 | .171 | .750 | .934 | .814 | 2 | 3.7 s |

`page recall@5` у всех локальных прогонов одинаковый — `.942` (retrieval
фиксирован), поэтому в таблице не дублируется. У DeepSeek reference на 20
кейсах page@5 `.800`.

## Agent, 20 кейсов

| модель | page | source | cit prec | faith | support | отказы | p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ministral 3 3B | .100 | .250 | .020 | .756 | .387 | 13 | 78.7 s |
| Gemma 3 4B | .350 | .400 | .088 | .841 | .751 | 11 | 53.5 s |
| Phi-4-mini | .400 | .750 | .101 | .840 | .486 | 0 | 31.1 s |
| **Qwen3-4B q4** | **.500** | **.800** | **.139** | **.849** | **.741** | 0 | 53.4 s |
| Qwen3-4B q8 | — | — | — | — | — | — | — |
| DeepSeek Flash ref | — | — | — | — | — | — | — |

## Выводы

1. **Qwen3-4B q4 — лучший локальный кандидат.** Меньше всего отказов (2/120),
   лучшие citation precision/recall (.530/.817), faithfulness .939 и support
   .868; в Agent — лучшие page/source recall (.500/.800) и нулевые отказы.
   p50 13.6 s в RAG и 53.4 s в Agent.
2. **Gemma 3 4B — второй.** 7 отказов, citation recall .717, support .788;
   в Agent support .751, но page recall ниже (.350) и p50 53.5 s.
3. **Phi-4-mini — быстрый, но нестабильный.** Самый низкий p50 в RAG
   (11.1 s), но 26 отказов и support .728; в Agent 0 отказов, однако support
   падает до .486.
4. **Ministral 3B — непригоден для этого пайплайна.** 57/120 отказов в RAG,
   13/20 в Agent, support .387, citation precision .292.
5. **Qwen3-4B q8 не проходит по железу.** Модель 5.7 ГБ, 25% слоёв ушло на
   CPU, скорость ~7.5 tok/s; первый же запрос упёрся в 2-минутный таймаут
   Ollama (7230 токенов). При 6 ГБ VRAM и `num_ctx=8192` практичный квант —
   **q4**, q8 требует больше VRAM или меньшего контекста.
6. **DeepSeek Flash — только reference.** Самый быстрый (3.7 s) и с хорошей
   faithfulness/support, но citation precision `.171` на 20 кейсах и облако;
   в прод не предлагается.

## Ограничения

- Судья `qwen3.5:4b` имеет human-holdout kappa `.456` (T103): метрики
  faithfulness/support — только relative ranking, не абсолютное качество.
- Один прогон на модель, без повторов и температуры; разброс не измерен.
- DeepSeek Flash — 20 кейсов, не полный набор; сравнение с локальными по
  citation precision ограничено.
- q8 не дал метрик: это вывод о несовместимости с 6 ГБ VRAM/таймаутом, а не
  оценка качества q8.
- q5/q6 Qwen3 не проверены; вывод о квантовании ограничен q4 vs q8.
