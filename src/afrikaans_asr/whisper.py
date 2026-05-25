from __future__ import annotations

from typing import Any

import torch
from transformers import WhisperProcessor


TORCH_DTYPES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def torch_dtype(name: str | None) -> torch.dtype:
    if name is None:
        return torch.float32
    normalized = str(name).lower()
    if normalized not in TORCH_DTYPES:
        raise ValueError(f"Unsupported torch dtype: {name}")
    return TORCH_DTYPES[normalized]


def load_processor(model_id: str, language: str, task: str) -> WhisperProcessor:
    return WhisperProcessor.from_pretrained(model_id, language=language, task=task)


def configure_generation(model: Any, processor: WhisperProcessor, language: str, task: str) -> None:
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=language,
        task=task,
    )
    model.generation_config.language = language
    model.generation_config.task = task
    model.generation_config.suppress_tokens = []
