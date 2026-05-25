from __future__ import annotations

import argparse

import torch
from transformers import WhisperForConditionalGeneration

from afrikaans_asr.config import load_config
from afrikaans_asr.datasets import dataset_size_summary, load_mapped_split
from afrikaans_asr.whisper import load_processor, torch_dtype


def run(config: dict) -> None:
    model_cfg = config["model"]
    data_cfg = config["data"]
    validation_cfg = config.get("validation", {})

    processor = load_processor(
        model_cfg["id"],
        model_cfg.get("language", "af"),
        model_cfg.get("task", "transcribe"),
    )
    train_ds = load_mapped_split(data_cfg["mapped_dataset"], data_cfg.get("train_split", "train"))
    val_ds = load_mapped_split(
        data_cfg["mapped_dataset"],
        data_cfg.get("validation_split", "validation"),
    )

    print(f"Train: {dataset_size_summary(train_ds)}")
    print(f"Validation: {dataset_size_summary(val_ds)}")

    sample = train_ds[int(data_cfg.get("sample_index", 0))]
    print(f"input_features type: {type(sample['input_features'])}")
    print(f"labels type: {type(sample['labels'])}")
    print(f"feature length: {len(sample['input_features'])}")
    print(f"label length: {len(sample['labels'])}")
    print("decoded label:")
    print(processor.tokenizer.decode(sample["labels"]))

    if not validation_cfg.get("run_forward_pass", True):
        return

    dtype = torch_dtype(validation_cfg.get("torch_dtype", "float32"))
    model = WhisperForConditionalGeneration.from_pretrained(model_cfg["id"], torch_dtype=dtype)
    model.eval()

    features = torch.tensor([sample["input_features"]], dtype=dtype)
    labels = torch.tensor([sample["labels"]])
    with torch.no_grad():
        outputs = model(input_features=features, labels=labels)
    print(f"Forward pass OK. Loss: {outputs.loss.item()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a mapped Afrikaans Whisper dataset.")
    parser.add_argument("--config", default="configs/validate.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
