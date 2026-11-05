"""NLI-проверка claim→citation на ONNX (mDeBERTa multilingual XNLI)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

DEFAULT_MODEL_DIR = Path("data/artifacts/models/mdeberta-nli")
MAX_LENGTH = 512


class NLICitationScorer:
    """Ленивая ONNX-модель: без torch, на onnxruntime + tokenizers."""

    def __init__(
        self,
        model_dir: Path = DEFAULT_MODEL_DIR,
        model_file: str = "onnx/model_quantized.onnx",
    ) -> None:
        from onnxruntime import InferenceSession
        from tokenizers import Tokenizer

        self.model_dir = Path(model_dir)
        config = json.loads(
            (self.model_dir / "config.json").read_text(encoding="utf-8")
        )
        labels = config.get("id2label", {})
        self._entailment = next(
            int(index)
            for index, label in labels.items()
            if str(label).casefold() == "entailment"
        )
        self._tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        self._tokenizer.enable_truncation(max_length=MAX_LENGTH)
        self._tokenizer.enable_padding()
        self._session = InferenceSession(
            str(self.model_dir / model_file),
            providers=["CPUExecutionProvider"],
        )

    def score(self, premise: str, hypothesis: str) -> float:
        return self.score_many([(premise, hypothesis)])[0]

    def score_many(self, pairs: Iterable[tuple[str, str]]) -> list[float]:
        pairs = list(pairs)
        if not pairs:
            return []
        encodings = self._tokenizer.encode_batch(
            [(premise[:2000], hypothesis[:1000]) for premise, hypothesis in pairs]
        )
        ids = np.array([encoding.ids for encoding in encodings], dtype=np.int64)
        mask = np.array(
            [encoding.attention_mask for encoding in encodings], dtype=np.int64
        )
        inputs = {"input_ids": ids, "attention_mask": mask}
        if any(item.name == "token_type_ids" for item in self._session.get_inputs()):
            inputs["token_type_ids"] = np.array(
                [encoding.type_ids for encoding in encodings], dtype=np.int64
            )
        logits = self._session.run(None, inputs)[0]
        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
        return [float(row[self._entailment]) for row in probabilities]

    def supports(
        self,
        claim: str,
        excerpts: Iterable[str],
        *,
        threshold: float = 0.5,
    ) -> bool:
        texts = [text for text in excerpts if text.strip()]
        if not texts:
            return False
        return max(self.score_many([(text, claim) for text in texts])) >= threshold


__all__ = ["DEFAULT_MODEL_DIR", "NLICitationScorer"]
