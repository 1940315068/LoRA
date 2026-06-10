import json
import copy
import os
import copy
from typing import Any, Dict
import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML config file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(obj: Dict[str, Any], path: str) -> None:
    """Save a dictionary as a YAML file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def save_json(obj: Dict[str, Any], path: str) -> None:
    """Save a dictionary as a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def ensure_dir(path: str) -> None:
    """Create a directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def apply_lora_rank_override(config, rank=None):
    """
    Apply LoRA rank override and automatically set experiment name/output_dir.
    """
    config = copy.deepcopy(config)

    if rank is not None:
        config["lora"]["r"] = rank
        config["lora"]["alpha"] = 2 * rank

    model_tag = config["model"].get("model_short_name", "model")
    method = config.get("experiment", {}).get("method", "lora")
    rank = config["lora"]["r"]

    experiment_name = f"{model_tag}_{method}_r{rank}"
    config["experiment_name"] = experiment_name

    model_output_dir = config.get("output", {}).get(
        "model_output_dir",
        f"src/outputs/{model_tag}",
    )

    config["training"]["output_dir"] = os.path.join(
        model_output_dir,
        f"{method}_r{rank}",
    )

    return config


def apply_model_override(config, model_key=None):
    config = copy.deepcopy(config)

    if model_key is None:
        model_key = config.get("model_key")

    if model_key is None:
        raise ValueError("model_key is required.")

    models = config.get("models", {})

    if model_key not in models:
        raise ValueError(
            f"Unknown model_key: {model_key}. "
            f"Available models: {list(models.keys())}"
        )

    model_info = models[model_key]

    config["model_key"] = model_key
    config.setdefault("model", {})
    config["model"]["model_name"] = model_info["model_name"]
    config["model"]["model_short_name"] = model_info["model_short_name"]

    config.setdefault("output", {})
    config["output"]["model_output_dir"] = (
        f"src/outputs/{model_info['model_short_name']}"
    )

    return config


def apply_method_override(config, method=None):
    config = copy.deepcopy(config)

    if method is None:
        method = config.get("experiment", {}).get("method", "lora")

    if method not in ["lora", "qlora"]:
        raise ValueError(f"Unsupported method: {method}")

    config.setdefault("experiment", {})
    config["experiment"]["method"] = method

    config.setdefault("quantization", {})

    if method == "qlora":
        config["quantization"]["load_in_4bit"] = True
    else:
        config["quantization"]["load_in_4bit"] = False

    return config