# NLI claim→citation: ONNX-эксперимент (T97, 2026-11-05)

## Модель

`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` — multilingual XNLI, русский
поддерживается; **ONNX quantized** (339 MB) + tokenizers, без torch.
Модель лежит в gitignored `data/artifacts/models/mdeberta-nli/`.

Скачивание:

```bash
uv run python - <<'PY'
from huggingface_hub import hf_hub_download
for f in ['config.json','tokenizer.json','tokenizer_config.json',
          'special_tokens_map.json','spm.model','onnx/model_quantized.onnx']:
    hf_hub_download('MoritzLaurer/mDeBERTa-v3-base-mnli-xnli', f,
                    local_dir='data/artifacts/models/mdeberta-nli')
PY
```

Метка `entailment` = 0, `neutral` = 1, `contradiction` = 2; score — softmax
вероятность entailment. Premise — предложения cited-excerpt (не весь чанк),
hypothesis — claim; support = max по предложениям ≥ threshold.

## Калибровка на human-cited (57) и holdout (50)

| набор | threshold | accuracy | precision | recall | kappa |
| --- | ---: | ---: | ---: | ---: | ---: |
| human-cited | .05 | .474 | .889 | .364 | **.123** |
| human-cited | .10 | .368 | .833 | .227 | .039 |
| holdout agent | .05 | .480 | **1.000** | .395 | **.155** |
| holdout agent | .10 | .380 | 1.000 | .279 | .098 |

Лучший threshold по kappa — **.05**, но даже он даёт слабое согласие:
`precision` высокий (NLI почти не подтверждает ложное), `recall` низкий
(теряет 28 из 44 human-true и 26 из 43 agent-true).

## Почему NLI проиграл

- Часть «true» — мета-утверждения и заголовки («Раздел 15.1 перечисляет…»,
  «Источники разделяют…»): NLI-модель не считает их entailment, хотя человек
  видит в cited-excerpt заголовок/перечень.
- Много парафраз и составных claim; модель строгая, а threshold ниже .05
  начинает ловить шум.
- Sentence-level premise не спас: long excerpt / meta claims остаются вне
  entailment.

## Вывод

NLI в этой конфигурации **не лучше LLM-судей** на cited-only:
human kappa .123 против .204 у strict_v2, holdout .155 против .623 у binary.
Этот результат — **negative**; в прод/NLI-гейт не включаем. Для следующей
попытки нужен либо более подходящий русский NLI, либо фильтрация
мета/заголовочных claim и отдельная разметка только фактических утверждений.

## Ограничения

- 57 human-cited и 50 agent-labeled — небольшие наборы; threshold .05 выбран
  по тем же 57, holdout только агентский.
- Модель квантованная ONNX, CPU; один threshold-свип, без повтора.
- `recall` низкий из-за meta/heading claim; это не доказывает, что NLI плох
  для чистых фактических утверждений.
