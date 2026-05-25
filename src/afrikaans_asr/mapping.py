from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from datasets import Dataset
from huggingface_hub import HfApi

from afrikaans_asr.config import load_config
from afrikaans_asr.datasets import dataset_size_summary, load_split_from_disk
from afrikaans_asr.whisper import load_processor


def prepare_dataset(
    dataset: Dataset,
    processor: Any,
    text_column: str,
    audio_column: str,
    max_label_length: int,
    num_proc: int,
    load_from_cache_file: bool,
) -> Dataset:
    def prepare_batch(batch: dict[str, Any]) -> dict[str, Any]:
        audio = batch[audio_column]
        batch["input_features"] = processor.feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
        ).input_features[0]
        batch["labels"] = processor.tokenizer(
            text_target=batch[text_column],
            truncation=True,
            max_length=max_label_length,
        ).input_ids
        return batch

    return dataset.map(
        prepare_batch,
        remove_columns=dataset.column_names,
        num_proc=num_proc,
        load_from_cache_file=load_from_cache_file,
    )


def map_split(
    split_name: str,
    dataset: Dataset,
    mapped_root: Path,
    processor: Any,
    config: dict[str, Any],
) -> None:
    target = mapped_root / split_name
    overwrite = bool(config["mapping"].get("overwrite", False))
    if target.exists() and not overwrite:
        print(f"Already mapped: {split_name} -> {target}")
        return
    if target.exists() and overwrite:
        shutil.rmtree(target)

    print(f"Mapping {split_name}: {dataset_size_summary(dataset)}")
    mapped = prepare_dataset(
        dataset=dataset,
        processor=processor,
        text_column=config["data"].get("text_column", "text"),
        audio_column=config["data"].get("audio_column", "audio"),
        max_label_length=int(config["mapping"].get("max_label_length", 225)),
        num_proc=int(config["mapping"].get("num_proc", 1)),
        load_from_cache_file=bool(config["mapping"].get("load_from_cache_file", False)),
    )
    mapped.save_to_disk(str(target))
    print(f"Saved {split_name}: {target}")


def upload_mapped_dataset(config: dict[str, Any], mapped_root: Path) -> None:
    hub = config["hub"]
    repo_id = hub["dataset_id"]
    api = HfApi()
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=bool(hub.get("private", False)),
        exist_ok=True,
    )
    api.upload_large_folder(
        folder_path=str(mapped_root),
        repo_id=repo_id,
        repo_type="dataset",
        num_workers=int(hub.get("num_workers", 8)),
    )
    print(f"Uploaded dataset: https://huggingface.co/datasets/{repo_id}")


def run(config: dict[str, Any], upload: bool = False) -> None:
    model = config["model"]
    data = config["data"]
    mapped_root = Path(data["mapped_root"])
    mapped_root.mkdir(parents=True, exist_ok=True)

    processor = load_processor(
        model["id"],
        model.get("language", "af"),
        model.get("task", "transcribe"),
    )
    train_ds = load_split_from_disk(data["raw_root"], data.get("train_split_dir", "train"))
    val_ds = load_split_from_disk(data["raw_root"], data.get("validation_split_dir", "val"))

    map_split(
        data.get("mapped_train_split_dir", "train"),
        train_ds,
        mapped_root,
        processor,
        config,
    )
    map_split(
        data.get("mapped_validation_split_dir", "validation"),
        val_ds,
        mapped_root,
        processor,
        config,
    )

    if upload:
        upload_mapped_dataset(config, mapped_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map raw Afrikaans ASR datasets for Whisper.")
    parser.add_argument("--config", default="configs/mapping.yaml")
    parser.add_argument("--upload", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(load_config(args.config), upload=args.upload)


if __name__ == "__main__":
    main()
