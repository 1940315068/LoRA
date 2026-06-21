import argparse
import heapq
import json
import os
from collections import Counter
from typing import Any, Dict, List, Tuple


EPS = 1e-12


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(
    data: Dict[str, Any],
    path: str,
) -> None:
    directory = os.path.dirname(path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def floor_to_multiple(
    value: float,
    step: int,
) -> int:
    if step <= 0:
        raise ValueError("rank_step must be positive.")

    return int(value // step) * step


def get_spectral_gain_for_step(
    module: Dict[str, Any],
    current_rank: int,
    rank_step: int,
    score_mode: str,
) -> float:
    """
    Compute the gain from increasing rank by rank_step.

    When current_rank=8 and rank_step=4, the gain is based on
    singular-value indices 8, 9, 10, and 11.
    """
    singular_values = module[
        "weighted_residual_singular_values"
    ]

    next_rank = min(
        current_rank + rank_step,
        len(singular_values),
    )

    if next_rank <= current_rank:
        return 0.0

    gain = sum(
        float(value) ** 2
        for value in singular_values[
            current_rank:next_rank
        ]
    )

    if score_mode == "absolute":
        return gain

    if score_mode == "output_normalized":
        denominator = float(
            module["weighted_output_sq"]
        )
        return gain / (denominator + EPS)

    if score_mode == "residual_normalized":
        denominator = float(
            module["weighted_error_sq"]
        )
        return gain / (denominator + EPS)

    raise ValueError(
        f"Unsupported score_mode: {score_mode}"
    )


def allocate_spectral_ranks(
    modules: List[Dict[str, Any]],
    avg_rank: int,
    min_rank: int,
    rank_step: int,
    max_rank_factor: float,
    score_mode: str,
) -> Tuple[Dict[str, int], Dict[str, Any]]:
    """
    Allocate adaptive ranks under the parameter budget of a uniform
    LoRA baseline.

    All allocated ranks are multiples of rank_step.

    max_rank is determined by:

        max_rank = floor(
            avg_rank * max_rank_factor / rank_step
        ) * rank_step

    Rank is increased by rank_step each time.
    """
    if not modules:
        raise ValueError(
            "No module results were found."
        )

    if rank_step not in {2, 4}:
        raise ValueError(
            "rank_step must be either 2 or 4."
        )

    if avg_rank % rank_step != 0:
        raise ValueError(
            f"avg_rank={avg_rank} must be a multiple "
            f"of rank_step={rank_step}."
        )

    if min_rank % rank_step != 0:
        raise ValueError(
            f"min_rank={min_rank} must be a multiple "
            f"of rank_step={rank_step}."
        )

    if max_rank_factor < 1.0:
        raise ValueError(
            "max_rank_factor must be at least 1.0."
        )

    max_rank = floor_to_multiple(
        avg_rank * max_rank_factor,
        rank_step,
    )

    max_rank = max(
        max_rank,
        avg_rank,
    )

    if min_rank > avg_rank:
        raise ValueError(
            "min_rank cannot be larger than avg_rank."
        )

    module_map = {
        module["module_name"]: module
        for module in modules
    }

    for module in modules:
        singular_values = module.get(
            "weighted_residual_singular_values",
            [],
        )

        if len(singular_values) < max_rank:
            raise ValueError(
                f"{module['module_name']} only has "
                f"{len(singular_values)} singular values, "
                f"but max_rank={max_rank}. "
                "Regenerate quantization_error.json with "
                f"--max_spectral_rank {max_rank} or larger."
            )

    # One LoRA rank for a matrix W in R^{out x in} adds:
    #
    #     in_dim + out_dim
    #
    # trainable parameters.
    rank_costs = {
        name: int(
            module["input_dim"]
            + module["output_dim"]
        )
        for name, module in module_map.items()
    }

    # Match the trainable-parameter budget of uniform rank avg_rank.
    target_parameter_budget = sum(
        avg_rank * rank_costs[name]
        for name in module_map
    )

    minimum_parameter_budget = sum(
        min_rank * rank_costs[name]
        for name in module_map
    )

    maximum_parameter_budget = sum(
        max_rank * rank_costs[name]
        for name in module_map
    )

    if target_parameter_budget < minimum_parameter_budget:
        raise ValueError(
            "Target budget is smaller than minimum-rank budget."
        )

    if target_parameter_budget > maximum_parameter_budget:
        raise ValueError(
            "Target budget is larger than maximum-rank budget."
        )

    ranks = {
        name: min_rank
        for name in module_map
    }

    current_parameter_count = minimum_parameter_budget

    # Heap item:
    #
    # (
    #   -gain_per_parameter,
    #   module_name,
    #   current_rank,
    # )
    heap = []

    def push_next_candidate(
        module_name: str,
    ) -> None:
        current_rank = ranks[module_name]
        next_rank = current_rank + rank_step

        if next_rank > max_rank:
            return

        module = module_map[module_name]

        gain = get_spectral_gain_for_step(
            module=module,
            current_rank=current_rank,
            rank_step=rank_step,
            score_mode=score_mode,
        )

        if gain <= 0:
            return

        step_parameter_cost = (
            rank_step * rank_costs[module_name]
        )

        gain_per_parameter = (
            gain / step_parameter_cost
        )

        heapq.heappush(
            heap,
            (
                -gain_per_parameter,
                module_name,
                current_rank,
            ),
        )

    for module_name in module_map:
        push_next_candidate(module_name)

    allocation_order = []

    while heap:
        (
            negative_utility,
            module_name,
            candidate_current_rank,
        ) = heapq.heappop(heap)

        # Ignore stale heap entries.
        if (
            candidate_current_rank
            != ranks[module_name]
        ):
            continue

        next_rank = (
            ranks[module_name]
            + rank_step
        )

        if next_rank > max_rank:
            continue

        step_parameter_cost = (
            rank_step
            * rank_costs[module_name]
        )

        if (
            current_parameter_count
            + step_parameter_cost
            > target_parameter_budget
        ):
            continue

        ranks[module_name] = next_rank

        current_parameter_count += (
            step_parameter_cost
        )

        allocation_order.append(
            {
                "module_name": module_name,
                "previous_rank": candidate_current_rank,
                "assigned_rank": next_rank,
                "rank_step": rank_step,
                "gain_per_parameter": float(
                    -negative_utility
                ),
            }
        )

        push_next_candidate(module_name)

    equivalent_average_rank = (
        current_parameter_count
        / sum(rank_costs.values())
    )

    diagnostics = {
        "avg_rank_target": avg_rank,
        "min_rank": min_rank,
        "max_rank": max_rank,
        "rank_step": rank_step,
        "max_rank_factor": max_rank_factor,

        "target_parameter_budget": int(
            target_parameter_budget
        ),
        "actual_parameter_count": int(
            current_parameter_count
        ),
        "unused_parameter_budget": int(
            target_parameter_budget
            - current_parameter_count
        ),
        "equivalent_average_rank": float(
            equivalent_average_rank
        ),

        "minimum_parameter_budget": int(
            minimum_parameter_budget
        ),
        "maximum_parameter_budget": int(
            maximum_parameter_budget
        ),

        "allocation_steps": len(
            allocation_order
        ),
        "allocation_order": allocation_order,
    }

    return ranks, diagnostics


def build_adaptive_rank_config(
    quant_error_report: Dict[str, Any],
    source_path: str,
    avg_rank: int,
    min_rank: int,
    rank_step: int,
    max_rank_factor: float,
    score_mode: str,
    alpha_ratio: float,
) -> Dict[str, Any]:
    modules = quant_error_report["modules"]

    rank_pattern, diagnostics = allocate_spectral_ranks(
        modules=modules,
        avg_rank=avg_rank,
        min_rank=min_rank,
        rank_step=rank_step,
        max_rank_factor=max_rank_factor,
        score_mode=score_mode,
    )
    max_rank = diagnostics["max_rank"]

    alpha_pattern = {
        module_name: int(
            round(alpha_ratio * rank)
        )
        for module_name, rank in rank_pattern.items()
    }

    rank_histogram = dict(
        sorted(
            Counter(
                rank_pattern.values()
            ).items()
        )
    )

    per_module = {}

    module_map = {
        module["module_name"]: module
        for module in modules
    }

    for module_name, rank in rank_pattern.items():
        module = module_map[module_name]

        singular_values = module[
            "weighted_residual_singular_values"
        ]

        selected_energy = sum(
            float(value) ** 2
            for value in singular_values[:rank]
        )

        total_error = float(
            module["weighted_error_sq"]
        )

        per_module[module_name] = {
            "rank": rank,
            "alpha": alpha_pattern[module_name],
            "layer_id": module["layer_id"],
            "projection": module["projection"],
            "input_dim": module["input_dim"],
            "output_dim": module["output_dim"],
            "activation_weighted_error": module[
                "activation_weighted_error"
            ],
            "selected_residual_energy_ratio": float(
                selected_energy
                / (total_error + EPS)
            ),
        }

    return {
        "method": "spectral_quantization_error_guided_qlora",

        "source_quantization_error_file": source_path,
        "model_name": quant_error_report["model_name"],
        "model_short_name": quant_error_report[
            "model_short_name"
        ],
        "dataset_name": quant_error_report[
            "dataset_name"
        ],

        "score_mode": score_mode,
        "budget_type": "lora_parameter_count",

        "avg_rank_target": avg_rank,
        "min_rank": min_rank,
        "max_rank": max_rank,
        "rank_step": rank_step,
        "max_rank_factor": max_rank_factor,
        "alpha_ratio": alpha_ratio,

        "rank_histogram": {
            str(rank): count
            for rank, count in rank_histogram.items()
        },

        "rank_pattern": rank_pattern,
        "alpha_pattern": alpha_pattern,

        "budget_diagnostics": diagnostics,
        "per_module": per_module,
    }


def print_summary(
    adaptive_config: Dict[str, Any],
) -> None:
    diagnostics = adaptive_config[
        "budget_diagnostics"
    ]

    ranks = list(
        adaptive_config["rank_pattern"].values()
    )

    print("=" * 100)
    print("Spectral quantization-error-guided rank allocation")
    print("-" * 100)
    print(
        f"Model: "
        f"{adaptive_config['model_short_name']}"
    )
    print(
        f"Score mode: "
        f"{adaptive_config['score_mode']}"
    )
    print(
        f"Target uniform rank: "
        f"{adaptive_config['avg_rank_target']}"
    )
    print(
        f"Allocated rank range: "
        f"{min(ranks)} -- {max(ranks)}"
    )
    print(
        f"Equivalent average rank: "
        f"{diagnostics['equivalent_average_rank']:.4f}"
    )
    print(
        f"Target parameter budget: "
        f"{diagnostics['target_parameter_budget']:,}"
    )
    print(
        f"Actual parameter count: "
        f"{diagnostics['actual_parameter_count']:,}"
    )
    print(
        f"Unused parameter budget: "
        f"{diagnostics['unused_parameter_budget']:,}"
    )
    print(
        f"Rank histogram: "
        f"{adaptive_config['rank_histogram']}"
    )
    print(
        f"Rank step: "
        f"{adaptive_config['rank_step']}"
    )

    print(
        f"Maximum rank factor: "
        f"{adaptive_config['max_rank_factor']}"
    )

    print(
        f"Maximum rank: "
        f"{adaptive_config['max_rank']}"
    )
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--quant_error",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--avg_rank",
        type=int,
        default=16,
        help=(
            "Parameter budget is matched to a uniform LoRA "
            "configuration with this rank."
        ),
    )

    parser.add_argument(
        "--min_rank",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--rank_step",
        type=int,
        default=4,
        choices=[2, 4],
        help="All adaptive ranks must be multiples of this value.",
    )

    parser.add_argument(
        "--max_rank_factor",
        type=float,
        default=2.0,
        help=(
            "Maximum rank relative to avg_rank. "
            "For avg_rank=16 and factor=2.0, max_rank=32."
        ),
    )

    parser.add_argument(
        "--score_mode",
        type=str,
        default="output_normalized",
        choices=[
            "absolute",
            "output_normalized",
            "residual_normalized",
        ],
    )

    parser.add_argument(
        "--alpha_ratio",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    report = load_json(args.quant_error)

    if args.output is None:
        output_dir = os.path.dirname(
            args.quant_error
        )

        output_path = os.path.join(
            output_dir,
            (
                f"adaptive_rank_spectral_"
                f"avg{args.avg_rank}.json"
            ),
        )
    else:
        output_path = args.output

    adaptive_config = build_adaptive_rank_config(
        quant_error_report=report,
        source_path=args.quant_error,
        avg_rank=args.avg_rank,
        min_rank=args.min_rank,
        rank_step=args.rank_step,
        max_rank_factor=args.max_rank_factor,
        score_mode=args.score_mode,
        alpha_ratio=args.alpha_ratio,
    )

    save_json(
        adaptive_config,
        output_path,
    )

    print_summary(adaptive_config)

    print(
        f"Saved adaptive rank config to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()