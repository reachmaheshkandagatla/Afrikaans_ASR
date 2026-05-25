from __future__ import annotations

import argparse

from huggingface_hub import HfApi


def run(folder: str, repo_id: str, repo_type: str) -> None:
    api = HfApi()
    api.upload_folder(folder_path=folder, repo_id=repo_id, repo_type=repo_type)
    print(f"Uploaded {folder} to {repo_type}: {repo_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a folder to the Hugging Face Hub.")
    parser.add_argument("--folder", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--repo-type", choices=["model", "dataset", "space"], default="model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.folder, args.repo_id, args.repo_type)


if __name__ == "__main__":
    main()
