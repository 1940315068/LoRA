import argparse
import os
import random
from typing import Dict
import time

import numpy as np
import torch
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from src.utils.dataset_utils import load_gsm8k_for_sft
from src.utils.io_utils import (
    ensure_dir,
    load_yaml,
    save_yaml,
    save_json,
    apply_model_override,
    apply_method_override,
    apply_lora_rank_override,
)
from src.utils.lora_utils import apply_lora, print_trainable_parameters, get_trainable_parameter_info
from src.utils.model_utils import load_model_and_tokenizer

def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenize_dataset(dataset, tokenizer, max_seq_length: int):
    """
    Tokenize text dataset for causal language modeling.
    """

    def tokenize_fn(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
    )

    return tokenized


def build_training_arguments(config: Dict) -> TrainingArguments:
    """
    Build Hugging Face TrainingArguments from YAML config.
    """
    train_cfg = config["training"]
    output_dir = train_cfg["output_dir"]

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_cfg["num_train_epochs"],
        learning_rate=train_cfg["learning_rate"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        logging_steps=train_cfg["logging_steps"],
        save_strategy=train_cfg.get("save_strategy", "epoch"),
        bf16=train_cfg.get("bf16", True),
        fp16=train_cfg.get("fp16", False),
        report_to="none",
        remove_unused_columns=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="Override LoRA rank r.",
    )
    parser.add_argument(
        "--model_key",
        type=str,
        default=None,
        help="Model key, e.g., qwen3_1p7b, qwen3_4b, qwen3_8b.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        choices=["lora", "qlora"],
        help="Fine-tuning method.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Override learning rate.",
    )
    
    args = parser.parse_args()

    config = load_yaml(args.config)
    config = apply_model_override(config, model_key=args.model_key)
    config = apply_method_override(config, method=args.method)
    config = apply_lora_rank_override(config, rank=args.rank)
    
    if args.learning_rate is not None:
        config["training"]["learning_rate"] = args.learning_rate

    seed = config["training"].get("seed", 42)
    set_seed(seed)

    output_dir = config["training"]["output_dir"]
    adapter_dir = os.path.join(output_dir, "adapter")

    ensure_dir(output_dir)

    print("=" * 80)
    print(f"Experiment: {config['experiment_name']}")
    print(f"Output dir: {output_dir}")
    print("=" * 80)

    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(config)

    print("Applying LoRA...")
    model = apply_lora(model, config)
    print_trainable_parameters(model)

    print("Loading dataset...")
    train_dataset = load_gsm8k_for_sft(config)

    print("Tokenizing dataset...")
    max_seq_length = config["training"]["max_seq_length"]
    tokenized_train_dataset = tokenize_dataset(
        train_dataset,
        tokenizer,
        max_seq_length=max_seq_length,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = build_training_arguments(config)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    print("Starting training...")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start_time = time.time()

    train_result = trainer.train()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    train_runtime = time.time() - start_time

    peak_memory_gb = None
    if torch.cuda.is_available():
        peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3

    train_metrics = train_result.metrics
    train_metrics["manual_train_runtime"] = train_runtime
    train_metrics["peak_memory_gb"] = peak_memory_gb

    param_info = get_trainable_parameter_info(model)
    train_metrics.update(param_info)

    trainer.save_metrics("train", train_metrics)

    print(f"Saving LoRA adapter to {adapter_dir}")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    config_used_path = os.path.join(output_dir, "config_used.yaml")
    save_yaml(config, config_used_path)

    print("Done.")
    print(f"Adapter saved to: {adapter_dir}")
    print(f"Config saved to: {config_used_path}")


if __name__ == "__main__":
    main()