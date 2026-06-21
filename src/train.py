import argparse
import os
import random
from typing import Dict
import time
import json

import numpy as np
import torch
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from src.utils.prompt_utils import build_qwen3_sft_text
from src.utils.dataset_utils import load_dataset_for_sft
from src.utils.io_utils import (
    ensure_dir,
    load_yaml,
    save_yaml,
    save_json,
    apply_model_override,
    apply_method_override,
    apply_lora_rank_override,
    apply_output_dir,
    infer_dataset_task,
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
        help="Override uniform LoRA rank. Only used by LoRA and QLoRA.",
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
        choices=["lora", "qlora", "adaqlora"],
        help="Fine-tuning method.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Override learning rate.",
    )
    parser.add_argument(
        "--adaptive_rank_config",
        type=str,
        default=None,
        help=(
            "Path to adaptive rank JSON. "
            "Required when --method adaqlora."
        ),
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional experiment directory name.",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load base config and apply model/method overrides
    # ------------------------------------------------------------------
    config = load_yaml(args.config)

    config = apply_model_override(
        config,
        model_key=args.model_key,
    )

    method = (
        args.method
        or config.get("experiment", {}).get("method")
        or "lora"
    )

    config = apply_method_override(
        config,
        method=method,
    )

    adaptive_rank_metadata = None

    # ------------------------------------------------------------------
    # 2. Configure rank
    # ------------------------------------------------------------------
    if method == "adaqlora":
        if args.adaptive_rank_config is None:
            raise ValueError(
                "--adaptive_rank_config is required when "
                "--method adaqlora."
            )

        if args.rank is not None:
            raise ValueError(
                "--rank should not be used with adaqlora. "
                "The ranks are loaded from --adaptive_rank_config."
            )

        if not os.path.isfile(args.adaptive_rank_config):
            raise FileNotFoundError(
                f"Adaptive rank config not found: "
                f"{args.adaptive_rank_config}"
            )

        with open(
            args.adaptive_rank_config,
            "r",
            encoding="utf-8",
        ) as f:
            adaptive_rank_metadata = json.load(f)

        rank_pattern = adaptive_rank_metadata.get(
            "rank_pattern"
        )
        alpha_pattern = adaptive_rank_metadata.get(
            "alpha_pattern"
        )

        if not rank_pattern:
            raise ValueError(
                "No rank_pattern found in adaptive rank config."
            )

        if not alpha_pattern:
            raise ValueError(
                "No alpha_pattern found in adaptive rank config."
            )

        if set(rank_pattern) != set(alpha_pattern):
            raise ValueError(
                "rank_pattern and alpha_pattern must contain "
                "the same module names."
            )

        # Validate and convert JSON values to integers.
        rank_pattern = {
            module_name: int(rank)
            for module_name, rank in rank_pattern.items()
        }

        alpha_pattern = {
            module_name: int(alpha)
            for module_name, alpha in alpha_pattern.items()
        }

        rank_step = int(
            adaptive_rank_metadata.get(
                "rank_step",
                1,
            )
        )

        for module_name, rank in rank_pattern.items():
            if rank <= 0:
                raise ValueError(
                    f"Invalid rank for {module_name}: {rank}"
                )

            if rank % rank_step != 0:
                raise ValueError(
                    f"Rank {rank} for {module_name} is not "
                    f"a multiple of rank_step={rank_step}."
                )

        avg_rank = int(
            adaptive_rank_metadata["avg_rank_target"]
        )

        alpha_ratio = float(
            adaptive_rank_metadata.get(
                "alpha_ratio",
                2.0,
            )
        )

        config.setdefault("lora", {})

        # Fallback values required by LoraConfig.
        # Actual module ranks are overridden by rank_pattern.
        config["lora"]["r"] = avg_rank
        config["lora"]["lora_alpha"] = int(
            round(avg_rank * alpha_ratio)
        )

        # Pass patterns to apply_lora().
        config["lora"]["rank_pattern"] = rank_pattern
        config["lora"]["alpha_pattern"] = alpha_pattern
        config["lora"]["adaptive_rank_config"] = (
            args.adaptive_rank_config
        )

        config.setdefault("experiment", {})
        config["experiment"]["method"] = "adaqlora"
        config["experiment"]["adaptive_rank"] = True
        config["experiment"]["avg_rank_target"] = avg_rank
        config["experiment"]["rank_step"] = rank_step
        config["experiment"]["max_rank"] = (
            adaptive_rank_metadata.get("max_rank")
        )
        config["experiment"]["score_mode"] = (
            adaptive_rank_metadata.get("score_mode")
        )

    else:
        if args.adaptive_rank_config is not None:
            raise ValueError(
                "--adaptive_rank_config can only be used with "
                "--method adaqlora."
            )

        config = apply_lora_rank_override(
            config,
            rank=args.rank,
        )

        config.setdefault("lora", {})
        config["lora"].pop("rank_pattern", None)
        config["lora"].pop("alpha_pattern", None)
        config["lora"]["adaptive_rank_config"] = None

        config.setdefault("experiment", {})
        config["experiment"]["adaptive_rank"] = False

    # ------------------------------------------------------------------
    # 3. Optional learning-rate override
    # ------------------------------------------------------------------
    if args.learning_rate is not None:
        config["training"]["learning_rate"] = (
            args.learning_rate
        )

    # ------------------------------------------------------------------
    # 4. Build experiment name and output directory
    # ------------------------------------------------------------------
    output_root = config.get(
        "output",
        {},
    ).get("model_output_dir")

    if output_root is None:
        raise KeyError(
            "config['output']['model_output_dir'] is required."
        )

    if args.run_name is not None:
        experiment_name = args.run_name

    elif method == "adaqlora":
        avg_rank = int(
            adaptive_rank_metadata["avg_rank_target"]
        )
        rank_step = int(
            adaptive_rank_metadata.get(
                "rank_step",
                1,
            )
        )
        max_rank = int(
            adaptive_rank_metadata.get(
                "max_rank",
                max(
                    adaptive_rank_metadata[
                        "rank_pattern"
                    ].values()
                ),
            )
        )
        score_mode = adaptive_rank_metadata.get(
            "score_mode",
            "spectral",
        )

        experiment_name = (
            f"adaqlora_avg{avg_rank}"
            f"_step{rank_step}"
            f"_max{max_rank}"
            f"_{score_mode}"
        )

    else:
        rank = int(config["lora"]["r"])
        experiment_name = f"{method}_r{rank}"

    output_dir = os.path.join(
        output_root,
        experiment_name,
    )

    config["experiment_name"] = experiment_name
    config["training"]["output_dir"] = output_dir

    # ------------------------------------------------------------------
    # 5. Reproducibility
    # ------------------------------------------------------------------
    seed = config["training"].get(
        "seed",
        42,
    )
    set_seed(seed)

    adapter_dir = os.path.join(
        output_dir,
        "adapter",
    )

    ensure_dir(output_dir)

    print("=" * 80)
    print(f"Method: {method}")
    print(f"Experiment: {experiment_name}")
    print(f"Output dir: {output_dir}")

    if adaptive_rank_metadata is not None:
        ranks = list(
            config["lora"]["rank_pattern"].values()
        )

        print(
            f"Adaptive rank range: "
            f"{min(ranks)} -- {max(ranks)}"
        )
        print(
            "Adaptive rank histogram: "
            f"{adaptive_rank_metadata.get('rank_histogram', {})}"
        )
        print(
            "Adaptive rank config: "
            f"{args.adaptive_rank_config}"
        )

    print("=" * 80)

    # ------------------------------------------------------------------
    # 6. Load base model
    # ------------------------------------------------------------------
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(
        config
    )

    # ------------------------------------------------------------------
    # 7. Apply uniform or adaptive LoRA
    # ------------------------------------------------------------------
    print(f"Applying {method.upper()}...")
    model = apply_lora(
        model,
        config,
    )

    print_trainable_parameters(model)

    # ------------------------------------------------------------------
    # 8. Load dataset
    # ------------------------------------------------------------------
    dataset_cfg = config.get(
        "dataset",
        {},
    )

    task_name = infer_dataset_task(config)
    dataset_name = dataset_cfg.get(
        "dataset_name",
        "unknown",
    )
    dataset_config = dataset_cfg.get(
        "dataset_config",
        None,
    )
    split = dataset_cfg.get(
        "split",
        "train",
    )

    print(
        f"Loading {task_name} dataset: "
        f"name={dataset_name}, "
        f"config={dataset_config}, "
        f"split={split}"
    )

    train_dataset = load_dataset_for_sft(
        config,
        tokenizer,
    )

    print(
        f"Loaded train examples: "
        f"{len(train_dataset)}"
    )

    # ------------------------------------------------------------------
    # 9. Tokenize
    # ------------------------------------------------------------------
    print("Tokenizing dataset...")

    max_seq_length = config["training"][
        "max_seq_length"
    ]

    tokenized_train_dataset = tokenize_dataset(
        train_dataset,
        tokenizer,
        max_seq_length=max_seq_length,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # ------------------------------------------------------------------
    # 10. Trainer
    # ------------------------------------------------------------------
    training_args = build_training_arguments(
        config
    )

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
        peak_memory_gb = (
            torch.cuda.max_memory_allocated()
            / 1024**3
        )

    train_metrics = train_result.metrics
    train_metrics["manual_train_runtime"] = (
        train_runtime
    )
    train_metrics["peak_memory_gb"] = (
        peak_memory_gb
    )
    train_metrics["method"] = method

    if adaptive_rank_metadata is not None:
        train_metrics["avg_rank_target"] = int(
            adaptive_rank_metadata[
                "avg_rank_target"
            ]
        )
        train_metrics["min_adaptive_rank"] = min(
            config["lora"][
                "rank_pattern"
            ].values()
        )
        train_metrics["max_adaptive_rank"] = max(
            config["lora"][
                "rank_pattern"
            ].values()
        )

    param_info = get_trainable_parameter_info(
        model
    )
    train_metrics.update(param_info)

    trainer.save_metrics(
        "train",
        train_metrics,
    )

    # ------------------------------------------------------------------
    # 11. Save adapter and config
    # ------------------------------------------------------------------
    print(
        f"Saving adapter to {adapter_dir}"
    )

    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    config_used_path = os.path.join(
        output_dir,
        "config_used.yaml",
    )
    save_yaml(
        config,
        config_used_path,
    )

    if adaptive_rank_metadata is not None:
        adaptive_config_used_path = os.path.join(
            output_dir,
            "adaptive_rank_used.json",
        )

        with open(
            adaptive_config_used_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                adaptive_rank_metadata,
                f,
                indent=2,
            )

    print("Done.")
    print(f"Adapter saved to: {adapter_dir}")
    print(f"Config saved to: {config_used_path}")


if __name__ == "__main__":
    main()