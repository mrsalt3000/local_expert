# Human-golden set для судьи (T95, 2026-11-04)

## Что подготовлено

Слепой пакет для человеческой разметки claim-level поддержки:

- `data/artifacts/t86-human/labels.csv` — **114 утверждений** из 19 ответов «Агента»,
  по 6 на ответ. **57 строк без цитат уже помечены `false`** (негативный
  контроль), **57 строк с цитатами оставлены пустыми** для вас.
- `data/artifacts/t86-human/packet.md` — те же утверждения в читаемом виде.
- `data/artifacts/t86-human/README.md` — правило разметки.
- `data/artifacts/t86-human/labels.template.json` — пустой шаблон.

В пакете **нет** вердиктов LLM-судьи и файлов `judges.json`; разметка должна
быть независимой. Полный текст книг и ответов лежит только в gitignored `data/artifacts/` и в git
не попадает.

`labels.csv` записан в **UTF-8 с BOM** и с разделителем `;` — Excel открывает
его без настройки кодировки. Скрипт принимает и `;`, и `,`, и `\t`.

## Правило

`true` — только если хотя бы один cited excerpt **прямо** подтверждает
утверждение. Excerpt по теме, частичный или с добавленными деталями — `false`.
Строки без цитат уже `false`: это проверка, не выдумывает ли судья поддержку
там, где цитат нет (DeepSeek в полном пакете дал 12 true на 75 таких строк).

## Как размечать

1. Открыть `labels.csv` и заполнить `label` только в **57 строках с непустым
   `cited`**; остальные 57 уже `false`.
2. Не открывать `judges.json`, `agreement.json` и отчёты судьи до окончания.
3. После заполнения запустить:

```bash
uv run python scripts/human_golden_packet.py \
  --packet data/artifacts/t86-calibration/packet.json \
  --output data/artifacts/t86-human \
  --labels data/artifacts/t86-human/labels.csv \
  --agent-labels data/artifacts/t86-calibration/labels.json \
  --judges data/artifacts/t86-calibration/deepseek data/artifacts/t86-calibration/ollama data/artifacts/t86-calibration/quote
```

Скрипт посчитает human↔agent agreement и human↔judge accuracy/precision/recall/
Cohen's kappa; результат — `data/artifacts/t86-human/agreement.json`.

## Ограничения

- Пакет собран из ответов 2026-11-04; если ответы/книги переиндексируются,
  id утверждений нужно перегенерировать.
- 114 утверждений — выборочные 6 из каждого ответа, а не все 586.
- Пока разметка не сделана, T86 остаётся частично закрытым: judge-профиль
  опирается на независимого агента, не на человека.

## Воспроизведение пакета

```bash
uv run python scripts/human_golden_packet.py \
  --packet data/artifacts/t86-calibration/packet.json \
  --output data/artifacts/t86-human --per-case 6
```
