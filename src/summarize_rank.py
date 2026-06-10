import argparse
import csv
import json
import os
from typing import Any, Dict, List, Optional


def load_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        print(f"[Warning] Missing file: {path}")
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_get(d: Optional[Dict[str, Any]], key: str, default=None):
    if d is None:
        return default
    return d.get(key, default)


def collect_effective_rank_result(
    output_root: str,
    method: str,
    rank: int,
) -> Optional[Dict[str, Any]]:
    path = os.path.join(
        output_root,
        f"{method}_r{rank}",
        "effective_rank.json",
    )

    report = load_json(path)

    if report is None:
        return None

    aggregate = report.get("aggregate", {})
    by_projection = report.get("by_projection", {})

    row = {
        "method": method,
        "rank": rank,
        "num_lora_modules": report.get("num_lora_modules"),
        "mean_rank_90": aggregate.get("mean_rank_90"),
        "mean_rank_95": aggregate.get("mean_rank_95"),
        "mean_rank_99": aggregate.get("mean_rank_99"),
        "min_rank_90": aggregate.get("min_rank_90"),
        "max_rank_90": aggregate.get("max_rank_90"),
        "min_rank_95": aggregate.get("min_rank_95"),
        "max_rank_95": aggregate.get("max_rank_95"),
        "min_rank_99": aggregate.get("min_rank_99"),
        "max_rank_99": aggregate.get("max_rank_99"),
        "mean_entropy_effective_rank": aggregate.get("mean_entropy_effective_rank"),
        "mean_normalized_entropy_effective_rank": aggregate.get(
            "mean_normalized_entropy_effective_rank"
        ),
        "mean_stable_rank": aggregate.get("mean_stable_rank"),
        "mean_numerical_rank": aggregate.get("mean_numerical_rank"),
    }

    for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        proj_summary = by_projection.get(proj, {})

        row[f"{proj}_mean_rank_90"] = proj_summary.get("mean_rank_90")
        row[f"{proj}_mean_rank_95"] = proj_summary.get("mean_rank_95")
        row[f"{proj}_mean_rank_99"] = proj_summary.get("mean_rank_99")
        row[f"{proj}_mean_entropy_effective_rank"] = proj_summary.get(
            "mean_entropy_effective_rank"
        )
        row[f"{proj}_mean_normalized_entropy_effective_rank"] = proj_summary.get(
            "mean_normalized_entropy_effective_rank"
        )
        row[f"{proj}_mean_stable_rank"] = proj_summary.get("mean_stable_rank")

    return row


def save_csv(rows: List[Dict[str, Any]], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = [
        "method",
        "rank",
        "num_lora_modules",
        "mean_rank_90",
        "mean_rank_95",
        "mean_rank_99",
        "min_rank_90",
        "max_rank_90",
        "min_rank_95",
        "max_rank_95",
        "min_rank_99",
        "max_rank_99",
        "mean_entropy_effective_rank",
        "mean_normalized_entropy_effective_rank",
        "mean_stable_rank",
        "mean_numerical_rank",
        "q_proj_mean_rank_90",
        "q_proj_mean_rank_95",
        "q_proj_mean_rank_99",
        "q_proj_mean_entropy_effective_rank",
        "q_proj_mean_normalized_entropy_effective_rank",
        "q_proj_mean_stable_rank",
        "k_proj_mean_rank_90",
        "k_proj_mean_rank_95",
        "k_proj_mean_rank_99",
        "k_proj_mean_entropy_effective_rank",
        "k_proj_mean_normalized_entropy_effective_rank",
        "k_proj_mean_stable_rank",
        "v_proj_mean_rank_90",
        "v_proj_mean_rank_95",
        "v_proj_mean_rank_99",
        "v_proj_mean_entropy_effective_rank",
        "v_proj_mean_normalized_entropy_effective_rank",
        "v_proj_mean_stable_rank",
        "o_proj_mean_rank_90",
        "o_proj_mean_rank_95",
        "o_proj_mean_rank_99",
        "o_proj_mean_entropy_effective_rank",
        "o_proj_mean_normalized_entropy_effective_rank",
        "o_proj_mean_stable_rank",
    ]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def fmt(value, digits: int = 2) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def print_table(rows: List[Dict[str, Any]]) -> None:
    print("=" * 100)
    print(
        f"{'Method':<8} "
        f"{'Rank':<6} "
        f"{'MeanR90':<10} "
        f"{'MeanR95':<10} "
        f"{'MeanR99':<10} "
        f"{'EntR':<10} "
        f"{'NormEntR':<10} "
        f"{'StableR':<10} "
        f"{'NumRank':<10}"
    )
    print("-" * 100)

    for row in rows:
        print(
            f"{row['method']:<8} "
            f"{str(row['rank']):<6} "
            f"{fmt(row['mean_rank_90']):<10} "
            f"{fmt(row['mean_rank_95']):<10} "
            f"{fmt(row['mean_rank_99']):<10} "
            f"{fmt(row['mean_entropy_effective_rank']):<10} "
            f"{fmt(row['mean_normalized_entropy_effective_rank']):<10} "
            f"{fmt(row['mean_stable_rank']):<10} "
            f"{fmt(row['mean_numerical_rank']):<10}"
        )

    print("=" * 100)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_root",
        type=str,
        default="src/outputs/qwen3_1p7b",
    )

    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["lora", "qlora"],
    )

    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=[4, 8, 16, 32, 64],
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    if args.output_csv is None:
        output_csv = os.path.join(
            args.output_root,
            "effective_rank_summary.csv",
        )
    else:
        output_csv = args.output_csv

    rows = []

    for method in args.methods:
        for rank in args.ranks:
            row = collect_effective_rank_result(
                output_root=args.output_root,
                method=method,
                rank=rank,
            )

            if row is not None:
                rows.append(row)

    save_csv(rows, output_csv)
    print_table(rows)

    print(f"Effective rank summary saved to: {output_csv}")


if __name__ == "__main__":
    main()