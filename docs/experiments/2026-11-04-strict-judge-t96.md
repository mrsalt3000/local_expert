# Строгий судья strict_v2 и калибровка на cited-only (T96, 2026-11-04)

## Что добавлено

`FaithfulnessJudge(strict_v2=True)` — opt-in режим, который вместо бинарного
вердикта требует:

- `verdict`: `full` / `partial` / `none`;
- `full` засчитывается только при наличии дословной `evidence`, найденной
  подстрокой в **cited**-фрагменте;
- заголовки, обрывки, мета-описания и утверждения с добавленными деталями —
  `partial`/`none`.

Флаг: `--strict-judge-v2` у `stage-eval`/`agent-eval`, в `meta` пишется
`judge_strict_v2: true` и `judge_mode: "strict_v2"`. По умолчанию выключен.

## Калибровка на human-cited (57 утверждений)

Те же human-метки T95, только строки с непустым `cited`:

| Судья | accuracy | precision | recall | kappa |
| --- | ---: | ---: | ---: | ---: |
| DeepSeek binary (default) | .7544 | .7679 | **.9773** | **−.0337** |
| DeepSeek strict (T79) | .7368 | .7959 | .8864 | .1355 |
| **DeepSeek strict_v2** | .6667 | **.8378** | .7045 | **.2043** |

`strict_v2` меняет профиль ошибок: ложных подтверждений меньше (fp 6 вместо
13), но появляется 13 пропусков верных утверждений. Accuracy падает, kappa
растёт с −.034 до **.204**; до «существенного» согласия (.6) далеко.

## Живой smoke

`stage-eval --limit 3 --faithfulness --judge-provider deepseek
--strict-judge-v2`:

- `meta.judge_mode = "strict_v2"`, `judge_strict_v2 = true`;
- `judged_cases 2`, `faithfulness 1.0`, `citation_support_precision 1.0`,
  ошибок судьи 0;
- отчёт `data/artifacts/t96-smoke.local.json`.

## Вывод

Строгий v2 — **улучшение, но не решение**: precision выше, kappa .204 против
−.034, однако recall падает до .705, а .204 всё ещё слабое согласие. Это
opt-in режим для аудита, не жёсткий citation-гейт. Для настоящего гейта нужен
семантический entailment/NLI или больше human-cited разметки; 4B/quote в
strict_v2 не прогонялись, поэтому сравнение с ними неполное.

## Ограничения

- Калибровка in-sample: промпт подбирался на тех же 57 утверждениях, на
  которых измерен; нужен holdout.
- Один прогон DeepSeek, без повтора на вариативность.
- 4B и quote-режимы в strict_v2 не измерены.
- `full` требует дословную цитату; парафраз, который человек считает
  поддержкой, может стать `partial` — это и есть 13 FN.

## Holdout-пакет (50 cited)

`data/artifacts/t96-holdout/` — **50 cited-утверждений из 20 новых кейсов**
(`combined-120`, offset 20; с T95-кейсами не пересекаются). Все строки с
непустым `cited`, негативный контроль уже проверен в T95.

- `labels.csv` — UTF-8 BOM, `;`, ID вида `h10_2` (Excel-safe), `label` пустая;
- `packet.md` / `README.md` — инструкция и читаемый пакет;
- `context.json` / `packet.json` — входы для судьи, в git не попадают.

После разметки:

```bash
uv run python scripts/eval_holdout_judge.py --provider deepseek
```

Скрипт прогонит `binary`, `strict` и `strict_v2` на одних и тех же 50 метках и
запишет `agreement.json`. Это **holdout**: промпт strict_v2 на этих
утверждениях не подбирался.

## Holdout-результат (agent-labeled, 50 cited)

Разметка независимым агентом (не человеком): **43 true / 7 false**.

| Судья | n | accuracy | precision | recall | kappa | tp/fp/fn/tn |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| binary | 50 | **.920** | .933 | **.977** | **.623** | 42/3/1/4 |
| strict | 50 | **.920** | .933 | **.977** | **.623** | 42/3/1/4 |
| strict_v2 | 50 | .840 | **1.000** | .814 | .551 | 35/0/8/7 |

- `strict` на этом holdout не изменил ни одного вердикта относительно binary.
- `strict_v2` **не дал ни одного ложного подтверждения** (fp 0), но пропустил 8 верных; precision 1.0, recall .814.
- По kappa default/strict лучше (.623 против .551), но если цель — не пропустить ни одного неподтверждённого claim, strict_v2 выигрывает по precision.

**Ограничение:** разметка агентская, не human; 43/50 true — высокая база, kappa чувствителен. Это не отменяет T95 human-вывод (−.034 на cited-only) и не делает strict_v2 гейтом.
