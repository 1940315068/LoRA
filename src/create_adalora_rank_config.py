"""
create_adalora_rank_config.py

Convert gradient importance scores (from compute_gradient_importance.py)
into a per-module LoRA rank allocation.

Algorithm
---------
Greedy marginal-gain allocation (identical to the AdaQLoRA pipeline):

  1. Start every module at min_rank.
  2. Maintain a total parameter budget = num_modules * avg_rank * cost_per_rank.
  3. Use a max-heap keyed on the marginal spectral gain of adding the next
     rank_step singular values to a module.
  4. Repeatedly increase the rank of the module with the highest marginal gain
     until the budget is exhausted or every module hits max_rank.

Score modes (passed to --score_mode):
  output_normalized   gain / weighted_output_sq   (default, mirrors AdaQLoRA)
  residual_normalized gain / weighted_error_sq
  absolute            raw squared singular value sum

Usage
-----
python -m src.create_adalora_rank_config \\
    --importance   src/outputs/qwen3_1p7b/gradient_importance.json \\
    --output       src/outputs/qwen3_1p7b/adalora_rank_avg16.json \\
    --avg_rank     16 \\
    --min_rank     2  \\
    --max_rank     32 \\
    --rank_step    2  \\
    --score_mode   output_normalized
"""

import argparse
import heapq
import json
import os
from collections import Counter
from typing import Any, Dict, List, Tuple


EPS = 1e-12


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def get_marginal_gain(
    module: Dict[str, Any],
    current_rank: int,
    rank_step: int,
    score_mode: str,
) -> float:
    """
    Marginal gain from increasing rank by rank_step, starting at current_rank.

    Uses weighted_residual_singular_values[current_rank : current_rank + rank_step].
    """
    svs = module["weighted_residual_singular_values"]
    next_rank = min(current_rank + rank_step, len(svs))

    if next_rank <= current_rank:
        return 0.0

    gain = sum(float(v) ** 2 for v in svs[current_rank:next_rank])

    if score_mode == "absolute":
        return gain
    if score_mode == "output_normalized":
        return gain / (float(module["weighted_output_sq"]) + EPS)
    if score_mode == "residual_normalized":
        return gain / (float(module["weighted_error_sq"]) + EPS)

    raise ValueError(f"Unsupported score_mode: {score_mode!r}")


# ---------------------------------------------------------------------------
# Greedy rank allocator
# ---------------------------------------------------------------------------

def allocate_ranks(
    modules: List[Dict[str, Any]],
    avg_rank: int,
    min_rank: int,
    max_rank: int,
    rank_step: int,
    score_mode: str,
    alpha_ratio: float = 2.0,
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, Any]]:
    """
    Greedy marginal-gain rank allocation.

    Returns
    -------
    rank_pattern  : {module_name: rank}
    alpha_pattern : {module_name: alpha}
    metadata      : summary statistics
    """
    if not modules:
        raise ValueError("No modules provided.")

    if min_rank > avg_rank:
        raise ValueError("min_rank cannot be larger than avg_rank.")

    if max_rank < avg_rank:
        raise ValueError("max_rank cannot be smaller than avg_rank.")

    for r in (min_rank, avg_rank, max_rank):
        if r % rank_step != 0:
            raise ValueError(
                f"rank={r} must be a multiple of rank_step={rank_step}."
            )

    # Clamp max_rank to the number of available singular values (set by
    # --max_spectral_rank in compute_gradient_importance).  Silently reduce
    # rather than crash so that high avg_rank runs still work.
    min_sv_count = min(len(m["weighted_residual_singular_values"]) for m in modules)
    if max_rank > min_sv_count:
        print(
            f"[Warning] max_rank={max_rank} exceeds available singular values "
            f"({min_sv_count}). Clamping max_rank to {min_sv_count}."
        )
        max_rank = min_sv_count
    # Snap max_rank down to nearest multiple of rank_step.
    max_rank = (max_rank // rank_step) * rank_step
    if max_rank < avg_rank:
        print(
            f"[Warning] After clamping, max_rank={max_rank} < avg_rank={avg_rank}. "
            f"Setting avg_rank=max_rank={max_rank}."
        )
        avg_rank = max_rank

    module_map = {m["module_name"]: m for m in modules}

    # Parameter cost per rank step for each module.
    rank_costs = {
        name: int(m["input_dim"] + m["output_dim"])
        for name, m in module_map.items()
    }

    # Budget = same total parameters as uniform avg_rank.
    target_budget = sum(avg_rank * rank_costs[n] for n in module_map)
    min_budget    = sum(min_rank * rank_costs[n] for n in module_map)
    max_budget    = sum(max_rank * rank_costs[n] for n in module_map)

    target_budget = max(min_budget, min(target_budget, max_budget))

    # Current rank per module (start at min_rank).
    current_ranks = {name: min_rank for name in module_map}
    used_budget   = sum(min_rank * rank_costs[n] for n in module_map)

    # Max-heap: (-score, name)
    heap = []
    for name, m in module_map.items():
        if current_ranks[name] < max_rank:
            score = get_marginal_gain(m, current_ranks[name], rank_step, score_mode)
            heapq.heappush(heap, (-score, name))

    # Greedy allocation.
    while heap and used_budget < target_budget:
        neg_score, name = heapq.heappop(heap)

        r = current_ranks[name]
        if r >= max_rank:
            continue

        cost_delta = rank_step * rank_costs[name]
        if used_budget + cost_delta > target_budget:
            continue  # Would overshoot budget; try next candidate.

        current_ranks[name] = r + rank_step
        used_budget += cost_delta

        if current_ranks[name] < max_rank:
            new_score = get_marginal_gain(
                module_map[name], current_ranks[name], rank_step, score_mode
            )
            heapq.heappush(heap, (-new_score, name))

    rank_pattern  = dict(current_ranks)
    alpha_pattern = {
        name: max(1, int(round(r * alpha_ratio)))
        for name, r in rank_pattern.items()
    }

    hist = dict(sorted(Counter(rank_pattern.values()).items()))

    metadata = {
        "avg_rank_target": avg_rank,
        "min_rank": min_rank,
        "max_rank": max_rank,
        "rank_step": rank_step,
        "score_mode": score_mode,
        "alpha_ratio": alpha_ratio,
        "actual_avg_rank": sum(rank_pattern.values()) / len(rank_pattern),
        "used_parameter_budget": used_budget,
        "target_parameter_budget": target_budget,
        "rank_histogram": hist,
    }

    return rank_pattern, alpha_pattern, metadata


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Create per-module LoRA rank config from gradient importance."
    )
    parser.add_argument(
        "--importance", type=str, required=True,
        help="Path to gradient_importance.json produced by compute_gradient_importance.py.",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output JSON path for the rank config.",
    )
    parser.add_argument(
        "--avg_rank", type=int, default=16,
        help="Average rank target (= total parameter budget of uniform r=avg_rank LoRA).",
    )
    parser.add_argument(
        "--min_rank", type=int, default=2,
        help="Minimum rank per module.",
    )
    parser.add_argument(
        "--max_rank", type=int, default=32,
        help="Maximum rank per module.",
    )
    parser.add_argument(
        "--rank_step", type=int, default=2,
        choices=[2, 4],
        help="Rank allocation granularity.",
    )
    parser.add_argument(
        "--score_mode", type=str, default="output_normalized",
        choices=["output_normalized", "residual_normalized", "absolute"],
        help="Gain normalization strategy.",
    )
    parser.add_argument(
        "--alpha_ratio", type=float, default=2.0,
        help="alpha = rank * alpha_ratio for each module.",
    )
    args = parser.parse_args()

    print("Loading importance scores...")
    report = load_json(args.importance)
    modules = report["modules"]
    print(f"  Model  : {report.get('model_name', 'unknown')}")
    print(f"  Method : {report.get('score_method', 'unknown')}")
    print(f"  Modules: {len(modules)}")

    print(
        f"\nAllocating ranks  avg={args.avg_rank}  min={args.min_rank}  "
        f"max={args.max_rank}  step={args.rank_step}  mode={args.score_mode}"
    )

    rank_pattern, alpha_pattern, meta = allocate_ranks(
        modules=modules,
        avg_rank=args.avg_rank,
        min_rank=args.min_rank,
        max_rank=args.max_rank,
        rank_step=args.rank_step,
        score_mode=args.score_mode,
        alpha_ratio=args.alpha_ratio,
    )

    output = {
        **meta,
        "model_name": report.get("model_name"),
        "score_method": report.get("score_method", "gradient_svd"),
        "rank_pattern": rank_pattern,
        "alpha_pattern": alpha_pattern,
    }

    save_json(output, args.output)

    print(f"\nRank histogram : {meta['rank_histogram']}")
    print(f"Actual avg rank: {meta['actual_avg_rank']:.2f}")
    print(f"Saved to       : {args.output}")


if __name__ == "__main__":
    main()
