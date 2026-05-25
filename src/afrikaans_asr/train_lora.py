from __future__ import annotations

import argparse
import inspect

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
)

from afrikaans_asr.collator import DataCollatorSpeechSeq2SeqWithPadding
from afrikaans_asr.config import load_config
from afrikaans_asr.datasets import load_mapped_split
from afrikaans_asr.metrics import build_wer_compute_metrics
from afrikaans_asr.whisper import configure_generation, load_processor, torch_dtype


def build_training_args(config: dict) -> Seq2SeqTrainingArguments:
    training = config["training"]
    hub = config.get("hub", {})
    args = {
        "output_dir": training["output_dir"],
        "per_device_train_batch_size": int(training.get("per_device_train_batch_size", 10)),
        "per_device_eval_batch_size": int(training.get("per_device_eval_batch_size", 8)),
        "gradient_accumulation_steps": int(training.get("gradient_accumulation_steps", 1)),
        "learning_rate": float(training.get("learning_rate", 2e-5)),
        "lr_scheduler_type": training.get("lr_scheduler_type", "cosine"),
        "warmup_ratio": float(training.get("warmup_ratio", 0.05)),
        "max_steps": int(training.get("max_steps", 6000)),
        "bf16": config["model"].get("torch_dtype", "bfloat16").lower()
        in {"bfloat16", "bf16"},
        "eval_steps": int(training.get("eval_steps", 1000)),
        "save_strategy": "steps",
        "save_steps": int(training.get("save_steps", 2000)),
        "save_total_limit": int(training.get("save_total_limit", 1)),
        "load_best_model_at_end": True,
        "metric_for_best_model": "wer",
        "greater_is_better": False,
        "predict_with_generate": True,
        "generation_max_length": int(training.get("generation_max_length", 225)),
        "generation_num_beams": int(training.get("generation_num_beams", 1)),
        "report_to": training.get("report_to", "none"),
        "gradient_checkpointing": bool(training.get("gradient_checkpointing", True)),
        "push_to_hub": bool(hub.get("push_to_hub", False)),
        "hub_model_id": hub.get("model_id"),
        "hub_private_repo": bool(hub.get("private", True)),
    }

    signature = inspect.signature(Seq2SeqTrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        args["eval_strategy"] = "steps"
    else:
        args["evaluation_strategy"] = "steps"

    return Seq2SeqTrainingArguments(**args)


def run(config: dict) -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model_cfg = config["model"]
    data_cfg = config["data"]
    lora_cfg = config["lora"]

    processor = load_processor(
        model_cfg["id"],
        model_cfg.get("language", "af"),
        model_cfg.get("task", "transcribe"),
    )
    train_ds = load_mapped_split(
        data_cfg["mapped_dataset"],
        data_cfg.get("train_split", "train"),
    )
    val_ds = load_mapped_split(
        data_cfg["mapped_dataset"],
        data_cfg.get("validation_split", "validation"),
    )

    dtype = torch_dtype(model_cfg.get("torch_dtype", "bfloat16"))
    model = WhisperForConditionalGeneration.from_pretrained(
        model_cfg["id"],
        device_map=model_cfg.get("device_map", "auto"),
        torch_dtype=dtype,
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=int(lora_cfg.get("r", 32)),
        lora_alpha=int(lora_cfg.get("lora_alpha", 64)),
        target_modules=list(
            lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "out_proj"])
        ),
        lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
        bias=lora_cfg.get("bias", "none"),
    )
    model = get_peft_model(model, peft_config)
    configure_generation(
        model,
        processor,
        model_cfg.get("language", "af"),
        model_cfg.get("task", "transcribe"),
    )
    model.print_trainable_parameters()

    trainer = Seq2SeqTrainer(
        model=model,
        args=build_training_args(config),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor, input_dtype=dtype),
        compute_metrics=build_wer_compute_metrics(processor),
    )

    trainer.train()

    if config.get("hub", {}).get("push_to_hub", False):
        trainer.push_to_hub()
        processor.push_to_hub(config["hub"]["model_id"])

    print("Training complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an Afrikaans Whisper LoRA adapter.")
    parser.add_argument("--config", default="configs/train_lora.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
