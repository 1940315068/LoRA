import json
from typing import Any, Dict, Optional, Tuple

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)


def build_lora_config(config: Dict) -> LoraConfig:
    """
    Build LoRA configuration from YAML config.
    """
    lora_cfg = config["lora"]

    adaptive_rank_path = lora_cfg.get(
        "adaptive_rank_config",
        None,
    )

    rank_pattern = {}
    alpha_pattern = {}

    if adaptive_rank_path is not None:
        adaptive_config = load_adaptive_rank_config(
            adaptive_rank_path
        )

        rank_pattern = {
            name: int(rank)
            for name, rank in adaptive_config[
                "rank_pattern"
            ].items()
        }

        alpha_pattern = {
            name: int(alpha)
            for name, alpha in adaptive_config[
                "alpha_pattern"
            ].items()
        }

        print(
            "Using adaptive rank configuration: "
            f"{adaptive_rank_path}"
        )
        print(
            "Adaptive rank histogram: "
            f"{adaptive_config.get('rank_histogram', {})}"
        )

    lora_config = LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(
            lora_cfg["lora_alpha"]
        ),
        lora_dropout=float(
            lora_cfg.get("lora_dropout", 0.0)
        ),
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg.get("bias", "none"),
        task_type="CAUSAL_LM",

        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
    )
    return lora_config


def apply_lora(model, config: Dict):
    """
    Attach LoRA adapters to the base model.
    """
    quant_cfg = config.get("quantization", {})
    load_in_4bit = quant_cfg.get("load_in_4bit", False)

    if load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

    lora_config = build_lora_config(config)
    model = get_peft_model(model, lora_config)

    return model


def print_trainable_parameters(model) -> None:
    """
    Print the number of trainable parameters.
    """
    model.print_trainable_parameters()


def get_trainable_parameter_info(model):
    trainable_params = 0
    total_params = 0

    for _, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params

    return {
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_ratio": trainable_params / total_params,
    }


def load_adaptive_rank_config(
    path: str,
) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if "rank_pattern" not in config:
        raise ValueError(
            f"No rank_pattern found in {path}"
        )

    if "alpha_pattern" not in config:
        raise ValueError(
            f"No alpha_pattern found in {path}"
        )

    return config