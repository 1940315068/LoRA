import argparse
import os
import subprocess
import sys
from typing import List


METHOD_TO_CONFIG = {
    "lora": "src/configs/lora_qwen3_1p7b.yaml",
    "qlora": "src/configs/qlora_qwen3_1p7b.yaml",
    "adalora": "src/configs/adalora_qwen3_1p7b.yaml",
}

# Default output root; should match model_output_dir in YAML configs.
OUTPUT_ROOT = "src/outputs/qwen3_1p7b"

# Sub-directory inside OUTPUT_ROOT for intermediate rank config JSONs.
RANK_CONFIGS_DIR = os.path.join(OUTPUT_ROOT, "rank_configs")


def run_command(cmd: List[str]) -> None:
    print("=" * 100)
    print("Running:", " ".join(cmd))
    print("=" * 100)
    subprocess.run(cmd, check=True)


def run_adalora_sweep(ranks: List[int], skip_train: bool, skip_eval: bool) -> None:
    """
    AdaLoRA sweep:
      1. Compute gradient importance once (skipped if JSON already exists).
      2. For each avg_rank:
         a. Generate per-module rank allocation JSON.
         b. Train with that allocation.
         c. Evaluate the saved adapter.
    """
    config_path = METHOD_TO_CONFIG["adalora"]
    importance_path = os.path.join(OUTPUT_ROOT, "gradient_importance.json")
    os.makedirs(RANK_CONFIGS_DIR, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Step 1: gradient importance (run once)                              #
    # ------------------------------------------------------------------ #
    if not os.path.exists(importance_path):
        run_command(
            [
                sys.executable, "-m", "src.compute_gradient_importance",
                "--config", config_path,
                "--output", importance_path,
            ]
        )
    else:
        print(f"[adalora] Gradient importance already exists: {importance_path}")

    # ------------------------------------------------------------------ #
    # Step 2: per-rank loop                                               #
    # ------------------------------------------------------------------ #
    for avg_rank in ranks:
        rank_cfg_path = os.path.join(RANK_CONFIGS_DIR, f"adalora_rank_avg{avg_rank}.json")
        run_name      = f"adalora_avg{avg_rank}_output_normalized"
        adapter_path  = os.path.join(OUTPUT_ROOT, run_name, "adapter")

        # 2a. Create rank allocation config (always regenerate).
        # max_rank is capped automatically inside create_adalora_rank_config
        # based on the number of singular values in the importance JSON.
        run_command(
            [
                sys.executable, "-m", "src.create_adalora_rank_config",
                "--importance", importance_path,
                "--output",     rank_cfg_path,
                "--avg_rank",   str(avg_rank),
                "--min_rank",   "2",
                "--max_rank",   str(avg_rank * 4),  # intentionally generous; will be clamped
                "--rank_step",  "2",
                "--score_mode", "output_normalized",
            ]
        )

        # 2b. Train.
        if not skip_train:
            run_command(
                [
                    sys.executable, "-m", "src.train",
                    "--config",               config_path,
                    "--adaptive_rank_config", rank_cfg_path,
                ]
            )

        # 2c. Evaluate.
        if not skip_eval:
            run_command(
                [
                    sys.executable, "-m", "src.evaluate",
                    "--config",  config_path,
                    "--adapter", adapter_path,
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["lora", "qlora", "adalora"],
        help="Methods to run.",
    )
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=[4, 8, 16, 32, 64],
        help="Ranks to sweep (used as avg_rank for adalora).",
    )
    parser.add_argument(
        "--skip_train",
        action="store_true",
        help="Skip training and run evaluation only.",
    )
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="Skip evaluation and run training only.",
    )

    args = parser.parse_args()

    invalid_methods = [m for m in args.methods if m not in METHOD_TO_CONFIG]
    if invalid_methods:
        raise ValueError(
            f"Unsupported methods: {invalid_methods}. Supported: {list(METHOD_TO_CONFIG)}"
        )

    for method in args.methods:
        if method == "adalora":
            run_adalora_sweep(args.ranks, args.skip_train, args.skip_eval)
            continue

        config_path = METHOD_TO_CONFIG[method]

        for rank in args.ranks:
            if not args.skip_train:
                run_command(
                    [
                        sys.executable, "-m", "src.train",
                        "--config", config_path,
                        "--rank",   str(rank),
                    ]
                )

            if not args.skip_eval:
                run_command(
                    [
                        sys.executable, "-m", "src.evaluate",
                        "--config", config_path,
                        "--rank",   str(rank),
                    ]
                )


if __name__ == "__main__":
    main()
