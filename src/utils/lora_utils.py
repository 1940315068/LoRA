from typing import Dict

from peft import (
    AdaLoraConfig,
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

    return LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def build_adalora_config(config: Dict) -> AdaLoraConfig:
    """
    Build AdaLoRA configuration from YAML config.
    """
    lora_cfg = config["lora"]
    adalora_cfg = config.get("adalora", {})

    target_r = int(lora_cfg["r"])
    default_init_r = max(target_r, 12)
    tinit = int(adalora_cfg.get("tinit", 200))
    tfinal = int(adalora_cfg.get("tfinal", 200))
    configured_total_step = int(adalora_cfg.get("total_step", 0) or 0)

    # PEFT validates total_step during config construction. Use a minimal
    # compatible placeholder and let train.py overwrite it with Trainer max_steps.
    min_valid_total_step = tinit + tfinal + 1
    total_step = (
        configured_total_step
        if configured_total_step > 0
        else max(min_valid_total_step, 1000)
    )

    return AdaLoraConfig(
        target_r=target_r,
        init_r=int(adalora_cfg.get("init_r", default_init_r)),
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        orth_reg_weight=float(adalora_cfg.get("orth_reg_weight", 0.5)),
        tinit=tinit,
        tfinal=tfinal,
        deltaT=int(adalora_cfg.get("deltaT", 10)),
        beta1=float(adalora_cfg.get("beta1", 0.85)),
        beta2=float(adalora_cfg.get("beta2", 0.85)),
        total_step=total_step,
        task_type=TaskType.CAUSAL_LM,
    )


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

    method = config.get("experiment", {}).get("method", "lora")

    if method == "adalora":
        lora_config = build_adalora_config(config)
    else:
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