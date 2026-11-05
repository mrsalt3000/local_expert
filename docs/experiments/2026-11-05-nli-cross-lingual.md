# Cross-lingual NLI: русский claim + английский excerpt (T100, 2026-11-05)

## Что измеряем

T95/T96/T99 проверяли только пары ru→ru: русское утверждение и русский
фрагмент. В каталоге есть английские книги, а ответы «Агента» всегда русские,
поэтому реальный cross-lingual случай — **английский cited-excerpt → русский
claim**. Здесь измеряется, тянет ли его ruBERT и тянет ли mDeBERTa.

## Корпус

- 1230 ready-книг; 19 книг детектор письменности считает английскими, ASCII-
  фильтр оставляет 15 с английской прозой, пригодные для пакета чанки есть
  у **12**.
- Остальные — mojibake (`Áåäíûé Ïàïà`), мусорные `Untitled`/`Microsoft Word -
  Document1` или служебные файлы без прозы.
- В пакет вошли 15 реальных фрагментов из этих 12 книг: AI Engineering,
  Agentic AI Playbook, Alice and Bob Learn Application Security, Database
  Reliability Engineering, Engineering RAG Systems, Hands-On RAG for
  Production, Lean Analytics, Radical Honesty (две книги), Release It!,
  The 7 Habits, The Developer's Playbook for LLM Security.

## Дизайн пакета

- На каждый excerpt — **два** русских claim: один прямо подтверждается
  фрагментом, второй похож по теме, но противоречит ему (заменено число,
  субъект или направление связи).
- 15 excerpt × 2 = **30 claim**; порядок перемешан seed 100, id `x01…x30`;
  excerpt — дословный текст chunk из каталога, язык проверен детектором
  (`en` для excerpt, `ru` для claim).
- `labels.csv` — слепая разметка, заполнена человеком: **17 true / 13 false**.
  Construction-метки лежат в `private/gold.local.json` и в пакет не входят.

## Метод

Как в T97–T99: premise — весь excerpt, score — максимум entailment по
цитируемым фрагментам; пороги фиксированы заранее: **ruBERT `.3`**,
**mDeBERTa `.05`**. Дополнительно снят sweep `.05–.9`, но порог по нему
не выбирается.

## Результат на human-метках

| модель | threshold | accuracy | precision | recall | kappa | 95% CI kappa |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ruBERT | .30 | **.700** | .722 | .765 | **.384** | .032–.670 |
| mDeBERTa | .05 | .567 | .571 | .941 | **.020** | −.187–.220 |

- ruBERT: 13 TP, 8 TN, 5 FP, 4 FN.
- mDeBERTa при `.05` снова почти всё считает положительным: 16 TP, 1 TN,
  12 FP, 1 FN; kappa неотличим от нуля.
- Bootstrap CI широкий из-за n=30, но у mDeBERTa ноль попадает в центр
  интервала, у ruBERT — только в нижний край.

### Проверка разметки

Human-метки совпали с construction-метками в **28 из 30** (kappa `.867`).
Две строки, где человек поставил `true`, а excerpt прямо противоречит
claim: `x04` (перепутаны Алиса и её коллега) и `x10` (6 трлн вместо
60 трлн teraflops). Они не перезаписывают human-файл; для чувствительности
собран `labels.audited.csv` с этими двумя `false`:

| модель | threshold | accuracy | precision | recall | kappa |
| --- | ---: | ---: | ---: | ---: | ---: |
| ruBERT | .30 | .700 | .667 | .800 | **.400** |
| mDeBERTa | .05 | .500 | .500 | .933 | **.000** |

Вывод не меняется: обе версии дают ruBERT около `.4`, mDeBERTa около нуля.

### Sweep

- ruBERT: лучший kappa `.437` при threshold `.2` (accuracy `.733`,
  recall `.882`); на audited-метках лучший остаётся `.3` / `.400`.
- mDeBERTa: лучший kappa `.143` при `.3`, но recall падает до `.235`;
  это подгонка порога на 30 строках, а не рабочая калибровка.

## Разбор ошибок ruBERT

- FP на числах и сущностях: модель видит оба числа в excerpt и не проверяет,
  к какому субъекту они относятся (`x05`, `x18`, `x19`, `x26`, `x15`).
- FN на парафразе: верное утверждение без лексического совпадения с
  английским фрагментом получает низкий score (`x09`, `x29`, `x14`).
- `x10` — ошибка human-метки, а не модели: ruBERT дал `.235`, то есть
  фактически отклонил неверное «6 трлн».

Это те же слабости, что и на ru→ru: композициональность и парафраз, а не
«язык как таковой».

## Вывод

1. Для **en→ru** нужна **ruBERT с порогом `.3`**: kappa `.384` (audited
   `.400`) против `.020` у mDeBERTa. Гипотеза «cross-lingual → обязательно
   multilingual» **не подтвердилась**.
2. mDeBERTa при своём пороге `.05` для en→ru непригодна: 12 FP из 13
   отрицательных. Даже лучший sweep `.143` требует recall `.235`.
3. Для **en→en** измерения нет: ruBERT русскоязычная, поэтому единственный
   кандидат — mDeBERTa. Но продукт отвечает по-русски, и en→en-ветки в
   маршруте нет; отдельный пакет нужен только если появится английский
   ответ.
4. Исходный план T101 «en/cross → mDeBERTa» **опровергнут для en→ru**.
   Переключатель по языку excerpt не даёт выигрыша на основном сценарии;
   реализовывать его сейчас не нужно.
5. В citation-гейт NLI по-прежнему нельзя: human kappa `.400` < `.6`,
   максимум — аудит-режим.

## Ограничения

- n=30, 15 excerpt / 12 книг, paired-дизайн; CI kappa широкий.
- Claim'ы составлены агентом по тексту книг, человек размечал слепо; две
  метки расходятся с excerpt и вынесены в audited-срез.
- Все excerpt — проза; таблицы и код не отбирались.
- Порог ruBERT `.3` перенесён из T99 и на этих данных не подбирался.

## Артефакты и воспроизведение

`data/artifacts/cross-lingual-nli/`: `labels.csv` (human),
`labels.audited.csv` (чувствительность), `results-rubert.json`,
`results-mdeberta.json`, `results-*-audited.json`, `agreement.json`.

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
