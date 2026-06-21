import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import bitsandbytes.functional as bnbF

from src.utils.model_utils import get_torch_dtype


def get_module_by_name(model, module_name: str):
    module = model

    for part in module_name.split("."):
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)

    return module


def parse_layer_id(module_name: str) -> Optional[int]:
    match = re.search(r"layers\.(\d+)", module_name)
    if match is None:
        return None
    return int(match.group(1))


def parse_projection_name(module_name: str) -> str:
    return module_name.split(".")[-1]


def find_target_module_names(model, target_modules: List[str]) -> List[str]:
    names = []

    for name, module in model.named_modules():
        if not hasattr(module, "weight"):
            continue

        for target in target_modules:
            if name.endswith(f".{target}") or name == target:
                names.append(name)
                break

    return names


def get_dequantized_weight(module) -> torch.Tensor:
    """
    Dequantize a bitsandbytes 4-bit weight to its original 2D shape.
    """
    weight = module.weight

    quant_state = getattr(weight, "quant_state", None)

    if quant_state is not None:
        dequantized = bnbF.dequantize_4bit(
            weight.data,
            quant_state=quant_state,
        )
    else:
        # Fallback for normal, non-quantized linear layers.
        dequantized = weight.data

    return dequantized.detach().to(
        device="cpu",
        dtype=torch.float32,
    )


def load_full_precision_model_cpu(config: Dict[str, Any]):
    model_name = config["model"]["model_name"]
    dtype_name = config["model"].get("torch_dtype", "bfloat16")
    trust_remote_code = config["model"].get("trust_remote_code", True)

    torch_dtype = get_torch_dtype(dtype_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map={"": "cpu"},
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )

    model.eval()
    return model


def load_quantized_model(config: Dict[str, Any]):
    model_name = config["model"]["model_name"]
    trust_remote_code = config["model"].get("trust_remote_code", True)

    quant_cfg = config.get("quantization", {})
    compute_dtype = get_torch_dtype(
        quant_cfg.get("bnb_4bit_compute_dtype", "bfloat16")
    )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=quant_cfg.get(
            "bnb_4bit_use_double_quant", True
        ),
    )

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

    model.eval()
    return model


def extract_problem_text(example: Dict[str, Any]) -> str:
    candidate_fields = [
        "question",
        "problem",
        "prompt",
        "input",
        "text",
        "description",
        "nl_problem",
        "problem_text",
    ]

    for field in candidate_fields:
        if field in example and example[field] is not None:
            return str(example[field]).strip()

    raise ValueError(
        f"Cannot find problem text field. Available fields: {list(example.keys())}"
    )


def build_calibration_prompt(tokenizer, problem_text: str, config: Dict[str, Any]) -> str:
    instruction = config.get("calibration", {}).get(
        "instruction",
        "Solve the following problem step by step.",
    )

    user_content = f"{instruction}\n\n{problem_text}"

    messages = [
        {
            "role": "user",
            "content": user_content,
        }
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    return user_content


def load_calibration_prompts(
    config: Dict[str, Any],
    tokenizer,
    max_samples: int,
) -> List[str]:
    dataset_name = config["dataset"]["dataset_name"]
    dataset_config = config["dataset"].get("dataset_config", None)

    split = config.get("calibration", {}).get(
        "split",
        config["dataset"].get("split", "train"),
    )

    if dataset_config is None:
        dataset = load_dataset(dataset_name, split=split)
    else:
        dataset = load_dataset(dataset_name, dataset_config, split=split)

    if max_samples is not None:
        max_samples = min(max_samples, len(dataset))
        dataset = dataset.select(range(max_samples))

    prompts = []

    for example in dataset:
        problem_text = extract_problem_text(example)
        prompt = build_calibration_prompt(
            tokenizer=tokenizer,
            problem_text=problem_text,
            config=config,
        )
        prompts.append(prompt)

    return prompts


def collect_activation_second_moments(
    model,
    tokenizer,
    prompts: List[str],
    target_modules: List[str],
    max_seq_length: int = 512,
    max_tokens_per_sample: Optional[int] = 128,
) -> Dict[str, Dict[str, Any]]:
    """
    Collect E[x_i^2] for the input activation of each target module.

    This avoids storing full activations and gives a diagonal approximation of:

        E_l = ||X(W - W_q)^T||_F / ||XW^T||_F
    """
    module_names = find_target_module_names(model, target_modules)

    stats = {
        name: {
            "sum_x2": None,
            "num_tokens": 0,
        }
        for name in module_names
    }

    hooks = []

    def make_hook(module_name):
        def hook_fn(module, inputs, output):
            x = inputs[0].detach()

            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])
            elif x.ndim != 2:
                return

            if max_tokens_per_sample is not None and x.shape[0] > max_tokens_per_sample:
                x = x[-max_tokens_per_sample:]

            x = x.float().cpu()
            sum_x2 = torch.sum(x * x, dim=0)

            if stats[module_name]["sum_x2"] is None:
                stats[module_name]["sum_x2"] = sum_x2
            else:
                stats[module_name]["sum_x2"] += sum_x2

            stats[module_name]["num_tokens"] += x.shape[0]

        return hook_fn

    for name in module_names:
        module = get_module_by_name(model, name)
        hooks.append(module.register_forward_hook(make_hook(name)))

    device = next(model.parameters()).device

    with torch.no_grad():
        for prompt in tqdm(prompts, desc="Collecting activation statistics"):
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_length,
            )

            inputs = {k: v.to(device) for k, v in inputs.items()}

            model(**inputs, use_cache=False)

    for hook in hooks:
        hook.remove()

    return stats


def compute_activation_weighted_errors(
    fp_model,
    quant_model,
    activation_stats: Dict[str, Dict[str, Any]],
    target_modules: List[str],
    max_spectral_rank: int = 64,
    svd_niter: int = 2,
    svd_device: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Compute scalar quantization metrics and the spectrum of the
    activation-weighted quantization residual.

    For module l:

        Delta W_l = W_l - W_l^q

        R_l = Delta W_l diag(sqrt(E[x_l^2]))

    The singular values of R_l estimate the error reduction obtainable
    from a low-rank correction.
    """
    module_names = find_target_module_names(
        quant_model,
        target_modules,
    )

    results = []
    eps = 1e-12

    for module_name in tqdm(
        module_names,
        desc="Computing quantization errors",
    ):
        if module_name not in activation_stats:
            continue

        stat = activation_stats[module_name]

        if stat["sum_x2"] is None or stat["num_tokens"] == 0:
            continue

        fp_module = get_module_by_name(fp_model, module_name)
        quant_module = get_module_by_name(quant_model, module_name)

        w_fp = fp_module.weight.detach().float().cpu()
        w_q = get_dequantized_weight(quant_module)

        if (
            w_q.ndim == 2
            and w_q.T.shape == w_fp.shape
            and w_q.shape != w_fp.shape
        ):
            w_q = w_q.T

        if w_fp.shape != w_q.shape:
            print(
                f"[Warning] Shape mismatch for {module_name}: "
                f"fp={tuple(w_fp.shape)}, "
                f"quant={tuple(w_q.shape)}"
            )
            continue

        # Diagonal approximation to E[x x^T].
        a2 = (
            stat["sum_x2"].float()
            / float(stat["num_tokens"])
        )

        diff = w_fp - w_q

        raw_error_norm = torch.norm(diff, p="fro")
        raw_weight_norm = torch.norm(w_fp, p="fro")

        raw_relative_error = (
            raw_error_norm
            / (raw_weight_norm + eps)
        )

        # Equivalent to ||Delta W diag(sqrt(E[x^2]))||_F^2.
        diff_col_norm_sq = torch.sum(
            diff.square(),
            dim=0,
        )

        weight_col_norm_sq = torch.sum(
            w_fp.square(),
            dim=0,
        )

        weighted_error_sq = torch.sum(
            a2 * diff_col_norm_sq
        )

        weighted_output_sq = torch.sum(
            a2 * weight_col_norm_sq
        )

        activation_weighted_error = torch.sqrt(
            weighted_error_sq
            / (weighted_output_sq + eps)
        )

        # Activation-weighted residual:
        #
        # R = (W - Wq) diag(sqrt(E[x^2]))
        sqrt_a2 = torch.sqrt(
            torch.clamp(a2, min=0.0) + eps
        )

        weighted_residual = (
            diff * sqrt_a2.unsqueeze(0)
        )

        singular_values = compute_top_singular_values(
            matrix=weighted_residual,
            max_rank=max_spectral_rank,
            niter=svd_niter,
            device=svd_device,
        )

        singular_energy = singular_values.square()
        captured_energy = float(
            singular_energy.sum().item()
        )

        total_residual_energy = float(
            weighted_error_sq.item()
        )

        captured_energy_ratio = (
            captured_energy
            / (total_residual_energy + eps)
        )

        rank_90 = rank_at_energy_threshold(
            singular_values,
            total_residual_energy,
            0.90,
        )

        rank_95 = rank_at_energy_threshold(
            singular_values,
            total_residual_energy,
            0.95,
        )

        rank_99 = rank_at_energy_threshold(
            singular_values,
            total_residual_energy,
            0.99,
        )

        result = {
            "module_name": module_name,
            "layer_id": parse_layer_id(module_name),
            "projection": parse_projection_name(module_name),

            "input_dim": int(w_fp.shape[1]),
            "output_dim": int(w_fp.shape[0]),
            "rank_parameter_cost": int(
                w_fp.shape[0] + w_fp.shape[1]
            ),

            "num_activation_tokens": int(
                stat["num_tokens"]
            ),
            "activation_rms_mean": float(
                torch.sqrt(torch.mean(a2)).item()
            ),

            "raw_relative_error": float(
                raw_relative_error.item()
            ),
            "activation_weighted_error": float(
                activation_weighted_error.item()
            ),

            "weighted_error_sq": total_residual_energy,
            "weighted_output_sq": float(
                weighted_output_sq.item()
            ),

            "mse": float(
                torch.mean(diff.square()).item()
            ),
            "max_abs_error": float(
                torch.max(torch.abs(diff)).item()
            ),

            # New spectral information.
            "weighted_residual_singular_values": [
                float(value)
                for value in singular_values.tolist()
            ],
            "spectral_rank_computed": int(
                singular_values.numel()
            ),
            "spectral_captured_energy_ratio": float(
                captured_energy_ratio
            ),
            "spectral_rank_90": rank_90,
            "spectral_rank_95": rank_95,
            "spectral_rank_99": rank_99,
        }

        results.append(result)

        del weighted_residual
        del diff
        del w_fp
        del w_q

    return results


def compute_top_singular_values(
    matrix: torch.Tensor,
    max_rank: int = 64,
    niter: int = 2,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Compute the largest singular values of a matrix using randomized SVD.

    The input matrix is expected to be on CPU. The computation is performed
    on GPU when possible, one module at a time.
    """
    if matrix.ndim != 2:
        raise ValueError(
            f"Expected a 2D matrix, but got shape {tuple(matrix.shape)}."
        )

    q = min(
        max_rank,
        matrix.shape[0],
        matrix.shape[1],
    )

    if q <= 0:
        return torch.empty(0, dtype=torch.float32)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        work_matrix = matrix.to(
            device=device,
            dtype=torch.float32,
        )

        _, singular_values, _ = torch.svd_lowrank(
            work_matrix,
            q=q,
            niter=niter,
        )

        singular_values = torch.sort(
            singular_values,
            descending=True,
        ).values

        singular_values = singular_values.detach().cpu()

        del work_matrix

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return singular_values

    except RuntimeError as exc:
        # Fall back to CPU when randomized SVD runs out of GPU memory.
        if "out of memory" not in str(exc).lower():
            raise

        print(
            "[Warning] CUDA out of memory during spectral analysis. "
            "Falling back to CPU."
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        work_matrix = matrix.to(
            device="cpu",
            dtype=torch.float32,
        )

        _, singular_values, _ = torch.svd_lowrank(
            work_matrix,
            q=q,
            niter=niter,
        )

        return torch.sort(
            singular_values,
            descending=True,
        ).values.cpu()


def rank_at_energy_threshold(
    singular_values: torch.Tensor,
    total_energy: float,
    threshold: float,
) -> Optional[int]:
    """
    Return the smallest rank whose captured energy reaches the threshold.

    Returns None when the saved singular values do not capture enough energy.
    """
    if singular_values.numel() == 0 or total_energy <= 0:
        return None

    singular_energy = singular_values.square()
    cumulative_energy = torch.cumsum(singular_energy, dim=0)

    target = threshold * total_energy
    indices = torch.nonzero(
        cumulative_energy >= target,
        as_tuple=False,
    )

    if indices.numel() == 0:
        return None

    return int(indices[0].item() + 1)


def summarize_error_values(
    module_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Summarize metrics without further grouping.
    """
    if len(module_results) == 0:
        return {}

    summary = {}

    for key in [
        "activation_weighted_error",
        "raw_relative_error",
        "mse",
        "max_abs_error",
    ]:
        vals = [item[key] for item in module_results]

        summary[f"mean_{key}"] = float(sum(vals) / len(vals))
        summary[f"min_{key}"] = float(min(vals))
        summary[f"max_{key}"] = float(max(vals))

    return summary


def summarize_quantization_errors(
    module_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compute aggregate statistics and statistics grouped by projection.
    """
    if len(module_results) == 0:
        return {
            "aggregate": {},
            "by_projection": {},
        }

    aggregate = summarize_error_values(module_results)

    grouped = defaultdict(list)

    for result in module_results:
        grouped[result["projection"]].append(result)

    by_projection = {
        projection: summarize_error_values(results)
        for projection, results in grouped.items()
    }

    return {
        "aggregate": aggregate,
        "by_projection": by_projection,
    }


def build_quantization_error_report(
    config: Dict[str, Any],
    module_results: List[Dict[str, Any]],
    max_calibration_samples: int,
    max_spectral_rank: int,
    svd_niter: int,
) -> Dict[str, Any]:
    summary = summarize_quantization_errors(
        module_results
    )

    report = {
        "model_name": config["model"]["model_name"],
        "model_short_name": config["model"]["model_short_name"],
        "dataset_name": config["dataset"]["dataset_name"],

        "max_calibration_samples": max_calibration_samples,
        "target_modules": config["lora"]["target_modules"],
        "num_modules": len(module_results),

        "spectral_analysis": {
            "residual_definition": (
                "(W_fp - W_q) * sqrt(E[x^2])"
            ),
            "max_spectral_rank": max_spectral_rank,
            "svd_niter": svd_niter,
        },

        "aggregate": summary.get(
            "aggregate",
            {},
        ),
        "by_projection": summary.get(
            "by_projection",
            {},
        ),
        "modules": module_results,
    }

    return report


def save_json(data: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)