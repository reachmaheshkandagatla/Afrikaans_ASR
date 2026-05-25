from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset, load_from_disk


def load_split_from_disk(root: str | Path, split_dir: str) -> Dataset:
    return load_from_disk(str(Path(root) / split_dir))


def load_mapped_split(source: str, split: str) -> Dataset:
    path = Path(source)
    if path.exists():
        return load_from_disk(str(path / split))
    return load_dataset(source, split=split)


def dataset_size_summary(dataset: Any) -> str:
    return f"{len(dataset)} rows, columns={dataset.column_names}"
