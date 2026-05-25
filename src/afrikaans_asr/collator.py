from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    input_dtype: torch.dtype = torch.float32

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        batch["input_features"] = batch["input_features"].to(self.input_dtype)

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch
