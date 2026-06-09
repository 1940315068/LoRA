import argparse
import os

from src.utils.rank_analysis_utils import (
    analyze_adapter_effective_rank,
    save_effective_rank_report,
)


def print_summary(report):
    aggregate = report["aggregate"]

    print("=" * 100)
    print(f"Adapter: {report['adapter_dir']}")
    print(f"LoRA rank: {report['lora_r']}")
    print(f"LoRA alpha: {report['lora_alpha']}")
    print(f"Scaling: {report['scaling']}")
    print(f"Number of LoRA modules: {report['num_lora_modules']}")
    print("-" * 100)

    print(f"{'Metric':<50} {'Value':<20}")
    print("-" * 100)

    for key, value in aggregate.items():
        if isinstance(value, float):
            print(f"{key:<50} {value:<20.4f}")
        else:
            print(f"{key:<50} {value:<20}")

    print("-" * 100)
    print("By projection:")
    print("-" * 100)

    for projection, projection_summary in report["by_projection"].items():
        print(f"[{projection}]")
        for key, value in projection_summary.items():
            if isinstance(value, float):
                print(f"  {key:<48} {value:<20.4f}")
            else:
                print(f"  {key:<48} {value:<20}")

    print("=" * 100)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--adapter",
        type=str,
        required=True,
        help="Path to PEFT LoRA/QLoRA adapter directory.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save effective rank JSON report.",
    )

    parser.add_argument(
        "--energy_thresholds",
        type=float,
        nargs="+",
        default=[0.90, 0.95, 0.99],
        help="Energy thresholds for effective rank computation.",
    )

    args = parser.parse_args()

    adapter_dir = args.adapter

    if args.output is None:
        output_path = os.path.join(
            os.path.dirname(adapter_dir),
            "effective_rank.json",
        )
    else:
        output_path = args.output

    report = analyze_adapter_effective_rank(
        adapter_dir=adapter_dir,
        energy_thresholds=args.energy_thresholds,
    )

    save_effective_rank_report(report, output_path)

    print_summary(report)
    print(f"Saved effective rank report to: {output_path}")


if __name__ == "__main__":
    main()