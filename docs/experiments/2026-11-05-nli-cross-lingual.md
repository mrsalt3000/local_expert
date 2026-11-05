# Cross-lingual NLI: русский claim + английский excerpt (T100, 2026-11-05)

## Что измеряем

T95/T96/T99 проверяли только пары ru→ru: русское утверждение и русский
фрагмент. В каталоге есть английские книги, а ответы «Агента» всегда русские,
поэтому реальный cross-lingual случай — **английский cited-excerpt → русский
claim**. Здесь измеряется, тянет ли его ruBERT и тянет ли mDeBERTa.

## Корпус

- 1230 ready-книг; по письменности 23 выглядят английскими.
- Пригодны **12**: остальные — mojibake (`Áåäíûé Ïàïà`), мусорные
  `Untitled`/`Microsoft Word - Document1` или служебные файлы без прозы.
- В пакет вошли 15 реальных фрагментов из этих 12 книг: AI Engineering,
  Agentic AI Playbook, Alice and Bob Learn Application Security, Database
  Reliability Engineering, Engineering RAG Systems, Hands-On RAG for
  Production, Lean Analytics, Radical Honesty (две книги), Release It!,
  The 7 Habits, The Developer's Playbook for LLM Security.

## Дизайн пакета

- На каждый excerpt — **два** русских claim: один прямо подтверждается
  фрагментом, второй похож по теме, но противоречит ему (заменено число,
  субъект или направление связи).
- 15 excerpt × 2 = **30 claim**, ровно 15 true / 15 false.
- Порядок перемешан seed 100, id `x01…x30`; excerpt в пакете — дословный
  текст chunk из каталога, язык проверен детектором (`en` для excerpt,
  `ru` для claim).
- `labels.csv` — слепая разметка (колонка `label` пустая), construction-метки
  лежат в `private/gold.local.json` и в пакет не входят.

## Метод

Как в T97–T99: premise — весь excerpt, score — максимум entailment по
цитируемым фрагментам; пороги фиксированы заранее: **ruBERT `.3`**,
**mDeBERTa `.05`**. Дополнительно снят sweep `.05–.9`, но порог по нему
не выбирается.

## Предварительный результат (construction gold, не human)

| модель | threshold | accuracy | precision | recall | kappa | mean score true/false |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ruBERT | .30 | **.700** | .667 | .800 | **.400** | .590 / .335 |
| mDeBERTa | .05 | .500 | .500 | .933 | **.000** | .208 / .164 |

- ruBERT: 12 TP, 9 TN, 6 FP, 3 FN. Лучший threshold sweep — тот же `.3`.
- mDeBERTa при `.05` помечает почти всё положительным (14 из 15 false —
  FP); лучший kappa в sweep `.15` — `.200` при recall `.400`.
- Это **не human-результат**: claim'ы и метки составлены агентом по тексту
  книг, поэтому таблица проверяет только воспроизводимость пайплайна и
  даёт направление, а не качество.

## Разбор ошибок ruBERT

- FP на числах: модель видит оба числа в excerpt и не проверяет, к какому
  субъекту они относятся (`x05`, `x18`).
- FN на парафразе: верное утверждение без лексического совпадения с
  английским фрагментом получает низкий score (`x09`, `x29`).

То есть ruBERT переносит на en→ru ту же слабость, что и на ru→ru:
композициональность и парафраз, а не «язык как таковой».

## Предварительный вывод

1. Для **en→ru** ruBERT остаётся лучше mDeBERTa даже cross-lingual:
   kappa `.400` против `.000` при фиксированных порогах. Гипотеза
   «cross-lingual → обязательно multilingual» пока не подтверждается.
2. Для **en→en** измерения нет: ruBERT русскоязычная, поэтому
   единственный кандидат — mDeBERTa; число появится только после
   отдельного en→en пакета.
3. Переключатель моделей по языку excerpt **преждевременен**: human-метки
   ещё не собраны, а mDeBERTa при своём пороге ведёт себя как «всё true».

## Ограничения

- Construction gold, автор claim'ов — агент; human-разметка обязательна.
- 15 excerpt / 12 книг, paired-дизайн; один прогон CPU ONNX.
- Все excerpt — проза; таблицы и код в пакет не отбирались намеренно.
- Порог ruBERT `.3` перенесён из T99 и на этих данных не подбирался.

## Дальше

Заполнить `label` в `data/artifacts/cross-lingual-nli/labels.csv` и прогнать:

```bash
uv run python scripts/eval_nli.py --cross-lingual \
  --model-dir data/artifacts/models/rubert-nli \
  --model-file onnx/model.onnx --threshold 0.3 \
  --labels data/artifacts/cross-lingual-nli/labels.csv \
  --output data/artifacts/cross-lingual-nli/results-rubert.json

uv run python scripts/eval_nli.py --cross-lingual \
  --model-dir data/artifacts/models/mdeberta-nli \
  --model-file onnx/model_quantized.onnx --threshold 0.05 \
  --labels data/artifacts/cross-lingual-nli/labels.csv \
  --output data/artifacts/cross-lingual-nli/results-mdeberta.json
```
