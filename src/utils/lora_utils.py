from typing import Dict

from peft import LoraConfig, TaskType, get_peft_model


def build_lora_config(config: Dict) -> LoraConfig:
    """
    Build LoRA configuration from YAML config.
    """
    lora_cfg = config["lora"]

    return LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def apply_lora(model, config: Dict):
    """
    Attach LoRA adapters to the base model.
    """
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