from typing import Dict

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)


def build_lora_config(config: Dict) -> LoraConfig:
    """
    Build LoRA configuration from YAML config.

    Supports optional rank_pattern / alpha_pattern (for adalora method):
    if config["lora"]["rank_pattern"] is a non-empty dict, each module
    gets its own rank as produced by create_adalora_rank_config.py.
    """
    lora_cfg = config["lora"]

    rank_pattern  = lora_cfg.get("rank_pattern")  or {}
    alpha_pattern = lora_cfg.get("alpha_pattern") or {}

    return LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=lora_cfg["target_modules"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
    )


def apply_lora(model, config: Dict):
    """
    Attach LoRA adapters to the base model.

    Works for method = lora | qlora | adalora.
    AdaLoRA uses a standard LoraConfig with per-module rank_pattern
    (no PEFT AdaLoraConfig; no dynamic rank pruning during training).
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