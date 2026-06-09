from typing import Dict, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


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


def build_quantization_config(config: Dict):
    quant_cfg = config.get("quantization", {})
    load_in_4bit = quant_cfg.get("load_in_4bit", False)

    if not load_in_4bit:
        return None

    compute_dtype = get_torch_dtype(
        quant_cfg.get("bnb_4bit_compute_dtype", "bfloat16")
    )

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=quant_cfg.get(
            "bnb_4bit_use_double_quant", True
        ),
    )


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
    Load the base model for LoRA fine-tuning.
    """
    model_name = config["model"]["model_name"]
    dtype_name = config["model"].get("torch_dtype", "bfloat16")
    trust_remote_code = config["model"].get("trust_remote_code", True)

    torch_dtype = get_torch_dtype(dtype_name)
    quantization_config = build_quantization_config(config)

    if quantization_config is not None:
        if torch.cuda.is_available():
            device_map = {"": torch.cuda.current_device()}
        else:
            device_map = None

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )

    model.config.use_cache = False

    return model


def load_model_and_tokenizer(config: Dict) -> Tuple:
    """
    Load tokenizer and base model.
    """
    tokenizer = load_tokenizer(config)
    model = load_base_model(config)

    return model, tokenizer