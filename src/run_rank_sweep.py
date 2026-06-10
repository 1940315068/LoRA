import argparse
import subprocess
import sys
from typing import List


METHOD_TO_CONFIG = {
    "lora": "src/configs/lora_qwen3_1p7b.yaml",
    "qlora": "src/configs/qlora_qwen3_1p7b.yaml",
    "adalora": "src/configs/adalora_qwen3_1p7b.yaml",
}


def run_command(cmd: List[str]) -> None:
    print("=" * 100)
    print("Running:", " ".join(cmd))
    print("=" * 100)
    subprocess.run(cmd, check=True)


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
        help="Ranks to run.",
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
        config_path = METHOD_TO_CONFIG[method]

        for rank in args.ranks:
            if not args.skip_train:
                run_command(
                    [
                        sys.executable,
                        "-m",
                        "src.train",
                        "--config",
                        config_path,
                        "--rank",
                        str(rank),
                    ]
                )

            if not args.skip_eval:
                run_command(
                    [
                        sys.executable,
                        "-m",
                        "src.evaluate",
                        "--config",
                        config_path,
                        "--rank",
                        str(rank),
                    ]
                )


if __name__ == "__main__":
    main()
