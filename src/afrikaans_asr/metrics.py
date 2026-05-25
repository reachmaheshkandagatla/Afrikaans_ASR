from __future__ import annotations

from typing import Any

import evaluate
import numpy as np


def build_wer_compute_metrics(processor: Any):
    wer_metric = evaluate.load("wer")

    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]

        labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
        pred_str = processor.tokenizer.batch_decode(preds, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)

        return {"wer": wer_metric.compute(predictions=pred_str, references=label_str)}

    return compute_metrics
