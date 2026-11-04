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
