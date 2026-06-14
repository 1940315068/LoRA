import argparse
import csv
import json
import os
from typing import Any, Dict, List, Optional


def load_json(path: str) -> Optional[Dict[str, Any]]:
    """
    Load a JSON file if it exists.
    """
    if not os.path.exists(path):
        print(f"[Warning] Missing file: {path}")
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_get(d: Optional[Dict[str, Any]], key: str, default=None):
    """
    Safely get a value from a dict.
    """
    if d is None:
        return default
    return d.get(key, default)


def collect_base_result(output_root: str) -> Optional[Dict[str, Any]]:
    """
    Collect base model evaluation result.

    Expected path:
        src/outputs/qwen3_1p7b/base/eval_results.json
    """
    base_eval_path = os.path.join(output_root, "base", "eval_results.json")
    base_eval = load_json(base_eval_path)

    if base_eval is None:
        return None

    row = {
        "method": "base",
        "rank": "",
        "accuracy": safe_get(base_eval, "accuracy"),
        "generation_success_rate": safe_get(base_eval, "generation_success_rate"),
        "execution_success_rate": safe_get(base_eval, "execution_success_rate"),
        "correct": safe_get(base_eval, "correct"),
        "num_eval_samples": safe_get(base_eval, "num_eval_samples"),
        "train_loss": "",
        "train_runtime": "",
        "eval_runtime": safe_get(base_eval, "eval_runtime"),
        "avg_generation_time_per_sample": safe_get(
            base_eval, "avg_generation_time_per_sample"
        ),
        "train_peak_gpu_memory_gb": "",
        "eval_peak_gpu_memory_gb": safe_get(base_eval, "peak_gpu_memory_gb"),
        "trainable_params": 0,
        "total_params": "",
        "trainable_ratio": 0,
    }

    return row


def collect_method_result(output_root: str, method: str, rank: int) -> Dict[str, Any]:
    """
    Collect result for one method and one rank.

    Expected:
        src/outputs/qwen3_1p7b/lora_r16/train_results.json
        src/outputs/qwen3_1p7b/lora_r16/lora_eval_results.json

        src/outputs/qwen3_1p7b/qlora_r16/train_results.json
        src/outputs/qwen3_1p7b/qlora_r16/lora_eval_results.json
    """
    exp_dir = os.path.join(output_root, f"{method}_r{rank}")

    train_path = os.path.join(exp_dir, "train_results.json")
    eval_path = os.path.join(exp_dir, f"{method}_eval_results.json")
    if not os.path.exists(eval_path):
        eval_path = os.path.join(exp_dir, "lora_eval_results.json")

    train_result = load_json(train_path)
    eval_result = load_json(eval_path)

    row = {
        "method": method,
        "rank": rank,
        "accuracy": safe_get(eval_result, "accuracy"),
        "generation_success_rate": safe_get(eval_result, "generation_success_rate"),
        "execution_success_rate": safe_get(eval_result, "execution_success_rate"),
        "correct": safe_get(eval_result, "correct"),
        "num_eval_samples": safe_get(eval_result, "num_eval_samples"),
        "train_loss": safe_get(train_result, "train_loss"),
        "train_runtime": safe_get(train_result, "train_runtime"),
        "eval_runtime": safe_get(eval_result, "eval_runtime"),
        "avg_generation_time_per_sample": safe_get(
            eval_result, "avg_generation_time_per_sample"
        ),
        "train_peak_gpu_memory_gb": safe_get(train_result, "peak_memory_gb"),
        "eval_peak_gpu_memory_gb": safe_get(eval_result, "peak_gpu_memory_gb"),
        "trainable_params": safe_get(train_result, "trainable_params"),
        "total_params": safe_get(train_result, "total_params"),
        "trainable_ratio": safe_get(train_result, "trainable_ratio"),
    }

    return row


def save_csv(rows: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save rows to a CSV file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = [
        "method",
        "rank",
        "accuracy",
        "generation_success_rate",
        "execution_success_rate",
        "correct",
        "num_eval_samples",
        "train_loss",
        "train_runtime",
        "eval_runtime",
        "avg_generation_time_per_sample",
        "train_peak_gpu_memory_gb",
        "eval_peak_gpu_memory_gb",
        "trainable_params",
        "total_params",
        "trainable_ratio",
    ]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def format_float(value, digits: int = 4) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return "" if value is None else str(value)


def print_table(rows: List[Dict[str, Any]]) -> None:
    """
    Print a compact summary table in terminal.
    """
    print("=" * 140)
    print(
        f"{'Method':<8} "
        f"{'Rank':<6} "
        f"{'Acc':<10} "
        f"{'GenRate':<10} "
        f"{'ExecRate':<10} "
        f"{'TrainLoss':<12} "
        f"{'TrainTime':<12} "
        f"{'EvalTime':<12} "
        f"{'TrainMem':<12} "
        f"{'EvalMem':<12} "
        f"{'TrainableParams':<16}"
    )
    print("-" * 140)

    for row in rows:
        print(
            f"{row['method']:<8} "
            f"{str(row['rank']):<6} "
            f"{format_float(row['accuracy']):<10} "
            f"{format_float(row.get('generation_success_rate')):<10} "
            f"{format_float(row.get('execution_success_rate')):<10} "
            f"{format_float(row['train_loss']):<12} "
            f"{format_float(row['train_runtime'], 1):<12} "
            f"{format_float(row['eval_runtime'], 1):<12} "
            f"{format_float(row['train_peak_gpu_memory_gb'], 2):<12} "
            f"{format_float(row['eval_peak_gpu_memory_gb'], 2):<12} "
            f"{str(row['trainable_params']):<16}"
        )

    print("=" * 140)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_root",
        type=str,
        default="src/outputs/qwen3_1p7b",
        help="Root output directory for one base model.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["lora", "qlora"],
        help="Methods to summarize.",
    )
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=[2, 4, 8, 16, 32, 64],
        help="Ranks to summarize.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Path to save summary CSV.",
    )
    parser.add_argument(
        "--no_base",
        action="store_true",
        help="Do not include base model result.",
    )

    args = parser.parse_args()

    output_csv = args.output_csv
    if output_csv is None:
        output_csv = os.path.join(args.output_root, "summary.csv")

    rows = []

    if not args.no_base:
        base_row = collect_base_result(args.output_root)
        if base_row is not None:
            rows.append(base_row)

    for method in args.methods:
        for rank in args.ranks:
            row = collect_method_result(args.output_root, method, rank)
            rows.append(row)

    save_csv(rows, output_csv)
    print_table(rows)

    print(f"Summary saved to: {output_csv}")


if __name__ == "__main__":
    main()