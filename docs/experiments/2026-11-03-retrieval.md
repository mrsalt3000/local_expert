# Эксперименты retrieval, 2026-11-03 (T64–T67)

Все замеры — на `data/eval/multi-book-40.local.json` (69 кейсов, 10 книг),
`top_k=20`, `scope=candidates` (24 книги), метрика страниц — пара
`(book_id, pdf_page_index)`. Воспроизведение: `scripts/retrieval_ab.py`,
артефакты — `data/artifacts/*.local.json` (git-ignored).

## T64 — диагностика промахов

9 промахов из 69. У 6 из 9 ожидаемой книги нет в 24 кандидатах (и в 40);
у 3 книга есть, но страница ниже 20-го места. OR-prefix5 не возвращает
ни одну из 9 страниц; глобальный поиск находит 5/9 (dense 2, OR 3).
Oracle-запрос по словам самой страницы находит все 6 «недостижимых» —
значит, дело в формулировке запроса, а не в индексе.

## T65 — OR-prefix5 вторым каналом RRF

| вариант | recall@5 | recall@20 | MRR@20 |
| --- | --- | --- | --- |
| baseline | .7826 | .8696 | .6752 |
| OR weight 0.05 | .7826 | .8696 | .7111 |
| OR weight 0.10 | .7826 | .8696 | .7296 |

Recall не меняется, MRR растёт; канал шумный, поэтому в проде выключен
(`BOOK_EXPERT_LEXICAL_OR_WEIGHT=0.0`).

```bash
uv run python scripts/retrieval_ab.py data/eval/multi-book-40.local.json --variant or
```

## T66 — книги глобального dense в candidate scope

| вариант | recall@5 | recall@20 | MRR@20 | scope |
| --- | --- | --- | --- | --- |
| baseline | .7826 | .8696 | .6752 | 24.0 |
| + global dense books | .6957 | .8551 | .6009 | 33.8 |
| + global OR books | .7826 | .8696 | .6752 | 24.0 |
| + both | .6957 | .8551 | .6009 | 33.8 |

Negative: добавленные книги размывают RRF и вытесняют верные страницы;
ни один из 9 промахов не закрылся.

```bash
uv run python scripts/retrieval_ab.py data/eval/multi-book-40.local.json --variant scope
```

## T67 — LLM query expansion

`qwen3.5:4b` генерирует 5–8 терминов, расширенный запрос идёт в routing и
retrieval. На 9 промахах expansion вернул 3 страницы (включая Nike-слоган),
но на всех 69 кейсах:

| вариант | recall@5 | recall@20 | MRR@20 | p50 |
| --- | --- | --- | --- | --- |
| baseline | .7826 | .8696 | .6752 | — |
| expansion | .6957 | .7681 | .5944 | 1833 мс |

Negative: свободные термины размывают запрос. Возвращаться только к
контролируемому расширению (термины из заголовков/секций книги).

```bash
uv run python scripts/retrieval_ab.py data/eval/multi-book-40.local.json --variant expand
```
