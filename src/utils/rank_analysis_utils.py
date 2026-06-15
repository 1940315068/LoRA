import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch


def load_adapter_config(adapter_dir: str) -> Dict[str, Any]:
    config_path = os.path.join(adapter_dir, "adapter_config.json")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Cannot find adapter config: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_adapter_state_dict(adapter_dir: str) -> Dict[str, torch.Tensor]:
    safetensors_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    bin_path = os.path.join(adapter_dir, "adapter_model.bin")

    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file

        return load_file(safetensors_path)

    if os.path.exists(bin_path):
        return torch.load(bin_path, map_location="cpu")

    raise FileNotFoundError(
        f"Cannot find adapter weights in {adapter_dir}. "
        "Expected adapter_model.safetensors or adapter_model.bin."
    )


def get_lora_base_key(key: str, lora_name: str) -> Optional[str]:
    """
    Convert keys like:

    base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
    base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight

    into:

    base_model.model.model.layers.0.self_attn.q_proj
    """
    pattern = rf"^(.*)\.{re.escape(lora_name)}(?:\.default)?(?:\.weight)?$"
    match = re.match(pattern, key)

    if match is None:
        return None

    return match.group(1)


def pair_lora_matrices(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, Dict[str, torch.Tensor]]:
    pairs = defaultdict(dict)

    for key, value in state_dict.items():
        if "lora_A" in key:
            base_key = get_lora_base_key(key, "lora_A")
            if base_key is not None:
                pairs[base_key]["A"] = value

        elif "lora_B" in key:
            base_key = get_lora_base_key(key, "lora_B")
            if base_key is not None:
                pairs[base_key]["B"] = value

        elif "lora_E" in key:
            base_key = get_lora_base_key(key, "lora_E")
            if base_key is not None:
                pairs[base_key]["E"] = value

    complete_pairs = {}

    for base_key, matrices in pairs.items():
        if "A" in matrices and "B" in matrices:
            complete_pairs[base_key] = matrices

    return complete_pairs


def parse_layer_id(module_name: str) -> Optional[int]:
    match = re.search(r"layers\.(\d+)", module_name)
    if match is None:
        return None
    return int(match.group(1))


def parse_projection_name(module_name: str) -> str:
    return module_name.split(".")[-1]


def compute_singular_values_from_lora(
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    scaling: float,
    lora_E: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    LoRA update:

        Delta W = scaling * B @ A

    A shape: [r, in_features]
    B shape: [out_features, r]

    Instead of forming the full Delta W matrix, compute its non-zero singular
    values using a small r x r matrix.

    B = Q_B R_B
    A^T = Q_A R_A
    B @ A = Q_B @ (R_B @ R_A^T) @ Q_A^T

    Therefore, the non-zero singular values of B @ A are the singular values of:

        R_B @ R_A^T
    """
    A = lora_A.detach().float().cpu()
    B = lora_B.detach().float().cpu()

    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"Expected 2D LoRA matrices, got A={A.shape}, B={B.shape}")

    if A.shape[0] != B.shape[1]:
        raise ValueError(f"Incompatible LoRA shapes: A={A.shape}, B={B.shape}")

    if lora_E is not None:
        E = lora_E.detach().float().cpu().reshape(-1)

        if E.numel() != A.shape[0]:
            raise ValueError(
                f"Incompatible AdaLoRA E shape: E={tuple(lora_E.shape)}, "
                f"expected rank={A.shape[0]}"
            )

        # AdaLoRA update: DeltaW = scaling * B @ diag(E) @ A
        B = B * E.unsqueeze(0)

    _, r_B = torch.linalg.qr(B, mode="reduced")
    _, r_A = torch.linalg.qr(A.T, mode="reduced")

    small_matrix = r_B @ r_A.T
    singular_values = torch.linalg.svdvals(small_matrix)

    singular_values = singular_values * abs(float(scaling))
    singular_values = torch.sort(singular_values, descending=True).values

    return singular_values


def compute_energy_rank(
    singular_values: torch.Tensor,
    threshold: float,
) -> int:
    if singular_values.numel() == 0:
        return 0

    energy = singular_values**2
    total_energy = energy.sum()

    if total_energy.item() == 0:
        return 0

    cumulative_energy = torch.cumsum(energy, dim=0)
    target = threshold * total_energy

    rank = int(torch.searchsorted(cumulative_energy, target).item()) + 1
    return rank


def compute_entropy_effective_rank(singular_values: torch.Tensor) -> float:
    if singular_values.numel() == 0:
        return 0.0

    energy = singular_values**2
    total_energy = energy.sum()

    if total_energy.item() == 0:
        return 0.0

    probs = energy / total_energy
    probs = probs[probs > 0]

    entropy = -torch.sum(probs * torch.log(probs))
    effective_rank = torch.exp(entropy)

    return float(effective_rank.item())


def compute_stable_rank(singular_values: torch.Tensor) -> float:
    if singular_values.numel() == 0:
        return 0.0

    max_sv = singular_values[0]

    if max_sv.item() == 0:
        return 0.0

    stable_rank = torch.sum(singular_values**2) / (max_sv**2)
    return float(stable_rank.item())


def compute_numerical_rank(singular_values: torch.Tensor, tol: float = 1e-6) -> int:
    if singular_values.numel() == 0:
        return 0

    max_sv = singular_values[0].item()

    if max_sv == 0:
        return 0

    return int(torch.sum(singular_values > tol * max_sv).item())


def analyze_lora_pair(
    module_name: str,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    scaling: float,
    energy_thresholds: List[float],
    lora_E: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    singular_values = compute_singular_values_from_lora(
        lora_A=lora_A,
        lora_B=lora_B,
        scaling=scaling,
        lora_E=lora_E,
    )

    nominal_rank = int(lora_A.shape[0])
    entropy_effective_rank = compute_entropy_effective_rank(singular_values)
    normalized_entropy_effective_rank = (
        entropy_effective_rank / nominal_rank if nominal_rank > 0 else 0.0
    )

    result = {
        "module_name": module_name,
        "layer_id": parse_layer_id(module_name),
        "projection": parse_projection_name(module_name),
        "shape_A": list(lora_A.shape),
        "shape_B": list(lora_B.shape),
        "shape_E": list(lora_E.shape) if lora_E is not None else None,
        "nominal_rank": nominal_rank,
        "scaling": float(scaling),
        "entropy_effective_rank": entropy_effective_rank,
        "normalized_entropy_effective_rank": normalized_entropy_effective_rank,
        "stable_rank": compute_stable_rank(singular_values),
        "numerical_rank": compute_numerical_rank(singular_values),
        "singular_values": [float(x) for x in singular_values.tolist()],
    }

    for threshold in energy_thresholds:
        key = f"rank_{int(threshold * 100)}"
        result[key] = compute_energy_rank(singular_values, threshold)

    return result


def summarize_module_results(
    module_results: List[Dict[str, Any]],
    energy_thresholds: List[float],
) -> Dict[str, Any]:
    if len(module_results) == 0:
        return {}

    summary = {
        "num_modules": len(module_results),
    }

    for threshold in energy_thresholds:
        key = f"rank_{int(threshold * 100)}"
        values = [m[key] for m in module_results]

        summary[f"mean_{key}"] = float(sum(values) / len(values))
        summary[f"min_{key}"] = int(min(values))
        summary[f"max_{key}"] = int(max(values))
        
    entropy_effective_ranks = [m["entropy_effective_rank"] for m in module_results]
    normalized_entropy_effective_ranks = [
        m["normalized_entropy_effective_rank"] for m in module_results
    ]
    stable_ranks = [m["stable_rank"] for m in module_results]
    numerical_ranks = [m["numerical_rank"] for m in module_results]
    
    summary["mean_entropy_effective_rank"] = float(
        sum(entropy_effective_ranks) / len(entropy_effective_ranks)
    )
    summary["min_entropy_effective_rank"] = float(min(entropy_effective_ranks))
    summary["max_entropy_effective_rank"] = float(max(entropy_effective_ranks))

    summary["mean_normalized_entropy_effective_rank"] = float(
        sum(normalized_entropy_effective_ranks)
        / len(normalized_entropy_effective_ranks)
    )
    summary["min_normalized_entropy_effective_rank"] = float(
        min(normalized_entropy_effective_ranks)
    )
    summary["max_normalized_entropy_effective_rank"] = float(
        max(normalized_entropy_effective_ranks)
    )
    
    summary["mean_stable_rank"] = float(sum(stable_ranks) / len(stable_ranks))
    summary["min_stable_rank"] = float(min(stable_ranks))
    summary["max_stable_rank"] = float(max(stable_ranks))

    summary["mean_numerical_rank"] = float(sum(numerical_ranks) / len(numerical_ranks))
    summary["min_numerical_rank"] = int(min(numerical_ranks))
    summary["max_numerical_rank"] = int(max(numerical_ranks))

    return summary


def summarize_by_projection(
    module_results: List[Dict[str, Any]],
    energy_thresholds: List[float],
) -> Dict[str, Any]:
    grouped = defaultdict(list)

    for result in module_results:
        grouped[result["projection"]].append(result)

    summary = {}

    for projection, results in grouped.items():
        summary[projection] = summarize_module_results(results, energy_thresholds)

    return summary


def analyze_adapter_effective_rank(
    adapter_dir: str,
    energy_thresholds: Optional[List[float]] = None,
) -> Dict[str, Any]:
    if energy_thresholds is None:
        energy_thresholds = [0.90, 0.95, 0.99]

    adapter_config = load_adapter_config(adapter_dir)
    state_dict = load_adapter_state_dict(adapter_dir)

    lora_r = int(adapter_config["r"])
    lora_alpha = float(adapter_config["lora_alpha"])
    scaling = lora_alpha / lora_r

    pairs = pair_lora_matrices(state_dict)

    if len(pairs) == 0:
        raise ValueError(f"No LoRA A/B matrix pairs found in adapter: {adapter_dir}")

    module_results = []

    for module_name, matrices in sorted(pairs.items()):
        result = analyze_lora_pair(
            module_name=module_name,
            lora_A=matrices["A"],
            lora_B=matrices["B"],
            scaling=scaling,
            energy_thresholds=energy_thresholds,
            lora_E=matrices.get("E"),
        )
        module_results.append(result)

    report = {
        "adapter_dir": adapter_dir,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "scaling": scaling,
        "energy_thresholds": energy_thresholds,
        "num_lora_modules": len(module_results),
        "aggregate": summarize_module_results(module_results, energy_thresholds),
        "by_projection": summarize_by_projection(module_results, energy_thresholds),
        "modules": module_results,
    }

    return report


def save_effective_rank_report(report: Dict[str, Any], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)