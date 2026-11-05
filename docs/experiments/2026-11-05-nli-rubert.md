# Русская NLI ruBERT на фактических claim (T99, 2026-11-05)

## Модель

`cointegrated/rubert-base-cased-nli-threeway` — русская three-way NLI
(entailment / contradiction / neutral), ONNX (`onnx/model.onnx`, 712 MB),
`token_type_ids` поддержан. Модель в gitignored
`data/artifacts/models/rubert-nli/`; `NLICitationScorer` получил
`model_file` и передачу `token_type_ids`.

Метки: entailment = 0, contradiction = 1, neutral = 2.

## Результаты на фактических claim

Порог выбран по human-cited; сравнение с mDeBERTa:

| набор | модель | threshold | accuracy | precision | recall | kappa |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| human factual 47 | mDeBERTa | .05 | .447 | .867 | .351 | .084 |
| human factual 47 | **ruBERT** | **.30** | **.787** | .829 | **.919** | **.256** |
| holdout factual 35 | mDeBERTa | .05 | .429 | 1.000 | .355 | .112 |
| holdout factual 35 | **ruBERT** | .30 | **.943** | **1.000** | **.935** | **.768** |
| holdout factual 35 | ruBERT | .15 | .971 | 1.000 | .968 | .873 |

На human-фактических kappa вырос с .084 до **.256**; recall .351 → .919.
На agent-holdout при том же .3 — .943 accuracy / kappa .768; при .15 — .873,
но это уже подгонка порога под holdout, поэтому основной кандидат — **.3**.

## Вывод

Русская NLI **материально лучше** multilingual mDeBERTa и впервые выглядит
рабочей: на фактических claim recall .92–.94 при precision .83–1.0. Но
human-kappa .256 всё ещё ниже «существенного» .6; holdout размечен агентом,
не человеком. Поэтому это **лучший NLI-кандидат**, но пока не citation-гейт.

Следующий шаг — human-разметка тех же 35 holdout-фактических claim и проверка
ruBERT на ней без изменения порога .3.

## Ограничения

- Human-фактических всего 47; порог .3 выбран на них же.
- Holdout — agent-labeled, не human.
- Модель 712 MB, CPU ONNX; один прогон.
- `NLICitationScorer` теперь поддерживает альтернативный `model_file` и
  `token_type_ids`, но product-путь по умолчанию остаётся mDeBERTa.
