# Afrikaans ASR Training Framework

This project turns the original three Colab notebooks into repeatable commands for:

1. mapping raw Afrikaans audio/text datasets into Whisper input features,
2. validating the mapped dataset on CPU,
3. fine-tuning `openai/whisper-large-v3` with LoRA on an A100-class GPU.

The defaults mirror the notebooks:

- base model: `openai/whisper-large-v3`
- language/task: Afrikaans transcription, `language="af"`, `task="transcribe"`
- mapped dataset repo: `rareRabbit/afrikaans-asr-mapped-largev3`
- LoRA model repo: `rareRabbit/whisper-largev3-afrikaans-lora`

## Install

For Colab or a local CUDA machine:

```bash
pip install -e .
```

If you need a specific PyTorch CUDA wheel, install PyTorch first from the official PyTorch
index, then run `pip install -e .`.

Log in to Hugging Face before upload or training with Hub push:

```bash
huggingface-cli login
```

## Phase 1: Map Dataset

Edit [configs/mapping.yaml](configs/mapping.yaml), then run:

```bash
afrikaans-asr-map --config configs/mapping.yaml
```

To upload the mapped folder to the dataset repo after mapping:

```bash
afrikaans-asr-map --config configs/mapping.yaml --upload
```

Expected raw dataset layout:

```text
datasets_general_v1/
  train/
  val/
```

Each split must contain an `audio` column with array/sampling rate and a transcript column
named `text`.

## Phase 2: Validate Mapped Dataset

Edit [configs/validate.yaml](configs/validate.yaml), then run:

```bash
afrikaans-asr-validate --config configs/validate.yaml
```

This loads one sample, checks feature and label shape, decodes labels with the matching
processor, and can run a tiny CPU forward pass.

## Phase 3: Train LoRA Adapter

Edit [configs/train_lora.yaml](configs/train_lora.yaml), then run on GPU:

```bash
afrikaans-asr-train --config configs/train_lora.yaml
```

The trainer saves checkpoints to `training.output_dir` and pushes the adapter plus processor
to `hub.model_id` when `hub.push_to_hub` is true.

## Upload Existing Output

To re-upload an existing local/Drive folder to a model or dataset repo:

```bash
afrikaans-asr-upload \
  --folder /path/to/folder \
  --repo-id rareRabbit/whisper-largev3-afrikaans-lora \
  --repo-type model
```

## Project Layout

```text
configs/                 YAML configs for each phase
src/afrikaans_asr/        reusable mapping, validation, training, upload code
scripts/                  small Colab helper snippets
```
