from typing import Dict, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_torch_dtype(dtype_name: str):
    """
    Convert dtype string in YAML to torch dtype.
    """
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32

    raise ValueError(f"Unsupported torch dtype: {dtype_name}")


def load_tokenizer(config: Dict):
    """
    Load tokenizer for the base model.
    """
    model_name = config["model"]["model_name"]
    trust_remote_code = config["model"].get("trust_remote_code", True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    return tokenizer


def load_base_model(config: Dict):
    """
    Load the full-precision or bf16 base model for LoRA fine-tuning.
    """
    model_name = config["model"]["model_name"]
    dtype_name = config["model"].get("torch_dtype", "bfloat16")
    trust_remote_code = config["model"].get("trust_remote_code", True)

    torch_dtype = get_torch_dtype(dtype_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )

    return model


def load_model_and_tokenizer(config: Dict) -> Tuple:
    """
    Load tokenizer and base model.
    """
    tokenizer = load_tokenizer(config)
    model = load_base_model(config)

    return model, tokenizer