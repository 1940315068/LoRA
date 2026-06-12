import json
import copy
import os
from typing import Any, Dict, Optional

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


# ============================================================
# Dataset / output path helpers
# ============================================================

def infer_dataset_task(config: Dict[str, Any]) -> str:
    """
    Infer dataset task from config.

    Preferred:
        dataset:
          task: math

    Fallback:
        infer from dataset.dataset_name.
    """
    dataset_cfg = config.get("dataset", {})
    task = dataset_cfg.get("task", None)

    if task is not None:
        return str(task).lower()

    dataset_name = str(dataset_cfg.get("dataset_name", "")).lower()

    if "gsm8k" in dataset_name:
        return "gsm8k"

    if "math" in dataset_name:
        return "math"

    raise ValueError("Cannot infer dataset task. Please set dataset.task.")


def get_model_output_dir(config: Dict[str, Any]) -> str:
    """
    Return the root output dir for a model + dataset.

    GSM8K:
        src/outputs/qwen3_1p7b

    MATH:
        src/outputs/qwen3_1p7b/math
    """
    model_short_name = config["model"]["model_short_name"]
    task_name = infer_dataset_task(config)

    if task_name == "gsm8k":
        return f"src/outputs/{model_short_name}"

    return f"src/outputs/{model_short_name}/{task_name}"


def get_experiment_output_dir(config: Dict[str, Any]) -> str:
    """
    Return output dir for LoRA/QLoRA experiment.

    Example:
        src/outputs/qwen3_1p7b/math/lora_r16
    """
    model_output_dir = get_model_output_dir(config)
    method = config.get("experiment", {}).get("method", "lora")
    rank = config["lora"]["r"]

    return os.path.join(model_output_dir, f"{method}_r{rank}")


def get_base_eval_output_dir(config: Dict[str, Any]) -> str:
    """
    Return output dir for base model evaluation.

    Example:
        src/outputs/qwen3_1p7b/math/base
    """
    return os.path.join(get_model_output_dir(config), "base")


def apply_output_dir(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply unified output dirs after model/method/rank overrides.

    This should be called after:
        apply_model_override
        apply_method_override
        apply_lora_rank_override
    """
    config = copy.deepcopy(config)

    config.setdefault("output", {})
    config.setdefault("training", {})

    config["output"]["model_output_dir"] = get_model_output_dir(config)
    config["training"]["output_dir"] = get_experiment_output_dir(config)

    return config


# ============================================================
# Config override helpers
# ============================================================

def apply_lora_rank_override(config, rank: Optional[int] = None):
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

    config.setdefault("output", {})
    config["output"]["model_output_dir"] = get_model_output_dir(config)
    config["training"]["output_dir"] = get_experiment_output_dir(config)

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
    config["output"]["model_output_dir"] = get_model_output_dir(config)

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