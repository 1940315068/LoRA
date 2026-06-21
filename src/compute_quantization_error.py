import argparse
import os

import torch

from src.utils.io_utils import load_yaml, apply_model_override
from src.utils.model_utils import load_tokenizer
from src.utils.quantization_error_utils import (
    load_full_precision_model_cpu,
    load_quantized_model,
    load_calibration_prompts,
    collect_activation_second_moments,
    compute_activation_weighted_errors,
    build_quantization_error_report,
    save_json,
)


def print_top_errors(report, top_k: int = 10):
    modules = report["modules"]
    modules = sorted(
        modules,
        key=lambda x: x["activation_weighted_error"],
        reverse=True,
    )

    print("=" * 100)
    print("Top quantization-sensitive modules")
    print("-" * 100)

    for item in modules[:top_k]:
        print(
            f"{item['module_name']:<65} "
            f"AWQE={item['activation_weighted_error']:.6f} "
            f"Raw={item['raw_relative_error']:.6f}"
        )

    print("=" * 100)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML.",
    )
    parser.add_argument(
        "--model_key",
        type=str,
        default=None,
        help="Model key, e.g., qwen3_1p7b, qwen3_4b, qwen3_8b.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Override output root, e.g., src/outputs/optmath/qwen3_8b.",
    )
    parser.add_argument(
        "--max_calibration_samples",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--max_tokens_per_sample",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--max_spectral_rank",
        type=int,
        default=64,
        help="Number of largest residual singular values to compute.",
    )
    parser.add_argument(
        "--svd_niter",
        type=int,
        default=2,
        help="Number of randomized SVD power iterations.",
    )
    parser.add_argument(
        "--svd_device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Device for spectral analysis. Default: CUDA when available.",
    )

    args = parser.parse_args()

    config = load_yaml(args.config)

    if args.model_key is not None and "models" in config:
        config = apply_model_override(config, model_key=args.model_key)

    config.setdefault("quantization", {})
    config["quantization"]["load_in_4bit"] = True

    if args.output_root is not None:
        config.setdefault("output", {})
        config["output"]["model_output_dir"] = args.output_root

    if args.output is None:
        output_dir = config["output"]["model_output_dir"]
        output_path = os.path.join(output_dir, "quantization_error.json")
    else:
        output_path = args.output

    print("=" * 100)
    print("Activation-weighted quantization error analysis")
    print(f"Model: {config['model']['model_name']}")
    print(f"Output: {output_path}")
    print("=" * 100)

    print("Loading tokenizer...")
    tokenizer = load_tokenizer(config)

    print("Loading calibration prompts...")
    prompts = load_calibration_prompts(
        config=config,
        tokenizer=tokenizer,
        max_samples=args.max_calibration_samples,
    )

    print(f"Number of calibration samples: {len(prompts)}")

    print("Loading full-precision model on CPU...")
    fp_model = load_full_precision_model_cpu(config)

    print("Loading 4-bit quantized model...")
    quant_model = load_quantized_model(config)

    print("Collecting activation second moments...")
    activation_stats = collect_activation_second_moments(
        model=quant_model,
        tokenizer=tokenizer,
        prompts=prompts,
        target_modules=config["lora"]["target_modules"],
        max_seq_length=args.max_seq_length,
        max_tokens_per_sample=args.max_tokens_per_sample,
    )

    print("Computing activation-weighted quantization errors...")
    module_results = compute_activation_weighted_errors(
        fp_model=fp_model,
        quant_model=quant_model,
        activation_stats=activation_stats,
        target_modules=config["lora"]["target_modules"],
        max_spectral_rank=args.max_spectral_rank,
        svd_niter=args.svd_niter,
        svd_device=args.svd_device,
    )

    report = build_quantization_error_report(
        config=config,
        module_results=module_results,
        max_calibration_samples=args.max_calibration_samples,
        max_spectral_rank=args.max_spectral_rank,
        svd_niter=args.svd_niter,
    )

    save_json(report, output_path)
    print_top_errors(report)

    print(f"Saved quantization error report to: {output_path}")

    del fp_model
    del quant_model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()