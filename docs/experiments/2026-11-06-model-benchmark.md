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
| DeepSeek Flash ref | 120 | **.539** | **.883** | **.941** | **.900** | 7 | **4.2 s** |

`page recall@5` одинаковый у всех, включая полный DeepSeek-прогон — `.942`
(retrieval фиксирован), поэтому в таблице не дублируется.

## Agent, 20 кейсов

| модель | page | source | cit prec | faith | support | отказы | p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ministral 3 3B | .100 | .250 | .020 | .756 | .387 | 13 | 78.7 s |
| Gemma 3 4B | .350 | .400 | .088 | .841 | .751 | 11 | 53.5 s |
| Phi-4-mini | .400 | .750 | .101 | .840 | .486 | 0 | 31.1 s |
| **Qwen3-4B q4** | **.500** | **.800** | **.139** | **.849** | **.741** | 0 | 53.4 s |
| Qwen3-4B q8 | — | — | — | — | — | — | — |
| DeepSeek Flash ref | .500 | .650 | .149 | .881 | .778 | 4 | **16.3 s** |

DeepSeek Flash Agent отвечает в среднем на **359.3 слова** (447.1 без
отказа); это cloud-reference, не кандидат в прод.

## Новые метрики

### Длина ответа, 20-кейсовый RAG-срез

| модель | avg words | avg words без отказов | отказы |
| --- | ---: | ---: | ---: |
| Qwen3-4B q4 | 173.8 | 182.5 | 1 |
| DeepSeek Flash (120) | 90.2 | 95.2 | 7 |
| Gemma 3 4B | 124.8 | 131.0 | 1 |
| Phi-4-mini | 116.4 | 143.5 | 4 |
| Ministral 3B | 63.1 | 191.8 | **14** |

У Ministral средняя занижена отказами: когда отвечает, он длинный (~192 слова).

### Agent-план при refine off

| модель | avg_queries | avg_rounds | avg_extra_queries |
| --- | ---: | ---: | ---: |
| Qwen3-4B q4 | 3.15 | 0.0 | n/a (refine off) |
| Gemma 3 4B | 4.00 | 0.0 | n/a (refine off) |
| Phi-4-mini | 4.00 | 0.0 | n/a (refine off) |
| Ministral 3B | 4.00 | 0.0 | n/a (refine off) |

`avg_rounds=0` и `extra=0` здесь — свойство конфигурации, а не модели;
сравнивать их в таком виде нельзя. Осмысленный замер — только refine-on A/B ниже.

## Refine-on A/B, 20 кейсов

| модель | refine | page | source | faith | support | queries | rounds | extra | p50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-4B q4 | off | .500 | .800 | .849 | .741 | 3.15 | 0.00 | 0.00 | 53.4 s |
| Qwen3-4B q4 | **on** | **.550** | .800 | **.877** | **.768** | 4.05 | **0.45** | **0.90** | 53.9 s |
| Gemma 3 4B | off | .350 | .400 | .841 | .751 | 4.00 | 0.00 | 0.00 | 53.5 s |
| Gemma 3 4B | **on** | .300 | .400 | **.856** | **.832** | 4.30 | **0.10** | **0.30** | **46.6 s** |

Refine добавляет **0.3–0.9 дополнительных запроса** и **0.1–0.45 раунда** на
запрос. У Qwen3-4B q4 это даёт небольшой рост page recall (+.05) и support
(+.027); у Gemma support растёт сильнее (+.081), но page recall падает
(−.05). Устойчивого выигрыша по retrieval нет, поэтому дефолт остаётся
`super_refine_enabled=false`; refine — opt-in для глубокого разбора, а не
режим сравнения моделей.

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
6. **DeepSeek Flash — сильный cloud-reference.** На полных 120 RAG-кейсах:
   citP **.539**, citR .883, faith .941, support **.900**, отказы 7,
   p50 **4.2 s**, 90.2 слова. Agent 20: page .500, source .650, support .778,
   p50 16.3 s. По RAG он конкурирует с Qwen3-4B q4 и быстрее, но это облако
   и внешняя зависимость; в прод по умолчанию не предлагается.
7. **Refine-on не даёт устойчивого выигрыша.** Он добавляет 0.3–0.9
   дополнительных запроса и 0.1–0.45 раунда; у Qwen3-4B q4 page +.05 и
   support +.027, у Gemma support +.081, но page −.05. Дефолт остаётся
   `refine=false`; refine — opt-in для глубокого разбора.

## Ограничения

- Судья `qwen3.5:4b` имеет human-holdout kappa `.456` (T103): метрики
  faithfulness/support — только relative ranking, не абсолютное качество.
- Один прогон на модель, без повторов и температуры; разброс не измерен.
- DeepSeek Flash — полный 120 RAG + 20 Agent; API был нестабилен (500/503),
  финальный прогон потребовал повторов. Это облако, не локальный кандидат.
- q8 не дал метрик: это вывод о несовместимости с 6 ГБ VRAM/таймаутом, а не
  оценка качества q8.
- q5/q6 Qwen3 не проверены; вывод о квантовании ограничен q4 vs q8.
