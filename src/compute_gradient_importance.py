"""
compute_gradient_importance.py

Compute gradient-based importance scores for adaptive LoRA rank allocation
WITHOUT quantization.

For each LoRA target module W in R^{out_dim x in_dim}:
  1. Run N calibration forward-backward passes on the task training data.
  2. Accumulate the gradient G = dL/dW across all passes.
  3. Compute a low-rank SVD of the mean gradient to obtain the per-rank
     importance signal (analogous to the residual singular values in AdaQLoRA).
  4. Record output activation energy ||Wx||^2 as the normalization baseline.

The output JSON mirrors quantization_error.json so that
create_adalora_rank_config.py (and the AdaQLoRA pipeline) can be reused
without modification.

Usage
-----
python -m src.compute_gradient_importance \\
    --config src/configs/adalora_qwen3_1p7b.yaml \\
    --output src/outputs/qwen3_1p7b/gradient_importance.json \\
    --max_calibration_samples 64 \\
    --max_spectral_rank 64
"""

import argparse
import json
import os
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

from src.utils.dataset_utils import load_gsm8k_for_sft
from src.utils.io_utils import load_yaml, ensure_dir
from src.utils.model_utils import load_model_and_tokenizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_target_linear_modules(
    model: nn.Module,
    target_suffixes: List[str],
) -> Dict[str, nn.Linear]:
    """
    Return all nn.Linear modules whose full name ends with one of
    target_suffixes, keyed by full module name.
    """
    result = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(name.endswith(t) for t in target_suffixes):
                result[name] = module
    return result


def _parse_layer_info(module_name: str):
    """Extract layer_id and projection from a full module name."""
    parts = module_name.split(".")
    projection = parts[-1]
    layer_id = None
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                layer_id = int(parts[i + 1])
            except ValueError:
                pass
            break
    return layer_id, projection


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_gradient_importance(
    model: nn.Module,
    target_modules: Dict[str, nn.Linear],
    dataloader: DataLoader,
    max_calibration_steps: int,
    max_spectral_rank: int,
    svd_niter: int = 4,
) -> List[Dict]:
    """
    Run calibration forward-backward passes and return per-module importance
    records in the same format as quantization_error.json.

    Parameters
    ----------
    model               : Full-precision model (no quantization).
    target_modules      : {name: nn.Linear} from get_target_linear_modules().
    dataloader          : Batched calibration DataLoader (batch_size=1 OK).
    max_calibration_steps: How many batches to process.
    max_spectral_rank   : Top-k singular values to keep.
    svd_niter           : Power iterations for torch.svd_lowrank.

    Returns
    -------
    List of dicts, one per module.
    """
    device = next(model.parameters()).device

    # Freeze all params; enable gradient only on target module weights.
    for param in model.parameters():
        param.requires_grad = False
    for module in target_modules.values():
        module.weight.requires_grad = True

    # Output-energy accumulators: sum of squared activations, token count.
    output_stats: Dict[str, Dict] = {
        name: {"sum_sq": 0.0, "num_tokens": 0}
        for name in target_modules
    }

    # Register forward hooks to capture output activation energy.
    hooks = []
    for name, module in target_modules.items():
        def _make_hook(n):
            def _hook(mod, inp, out):
                o = out.detach().float()
                output_stats[n]["sum_sq"] += float(o.pow(2).sum())
                # out shape: (batch, seq_len, out_dim) or (batch*seq, out_dim)
                output_stats[n]["num_tokens"] += o.numel() // o.shape[-1]
            return _hook
        hooks.append(module.register_forward_hook(_make_hook(name)))

    # -----------------------------------------------------------------------
    # Calibration passes
    # -----------------------------------------------------------------------
    model.train()
    steps_done = 0
    for batch in dataloader:
        if steps_done >= max_calibration_steps:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )
        outputs.loss.backward()
        steps_done += 1

    # Remove hooks.
    for h in hooks:
        h.remove()

    # -----------------------------------------------------------------------
    # Collect gradient SVD per module
    # -----------------------------------------------------------------------
    results = []
    for name, module in target_modules.items():
        grad = module.weight.grad
        if grad is None:
            print(f"[WARNING] No gradient for {name} — skipping.")
            continue

        # Mean gradient over calibration steps.
        G = grad.float() / max(steps_done, 1)
        out_dim, in_dim = G.shape

        # Low-rank SVD of the mean gradient.
        q = min(max_spectral_rank + 4, min(out_dim, in_dim))
        try:
            _, S, _ = torch.svd_lowrank(G, q=q, niter=svd_niter)
        except Exception:
            # Fallback to full SVD for very small matrices.
            S = torch.linalg.svdvals(G)

        singular_values = S[:max_spectral_rank].cpu().float().tolist()
        # Pad with zeros if matrix is smaller than max_spectral_rank.
        while len(singular_values) < max_spectral_rank:
            singular_values.append(0.0)

        stats = output_stats[name]
        num_tokens = max(stats["num_tokens"], 1)
        weighted_output_sq = stats["sum_sq"] / num_tokens
        weighted_error_sq = float(G.pow(2).sum())

        layer_id, projection = _parse_layer_info(name)

        results.append({
            "module_name": name,
            "layer_id": layer_id,
            "projection": projection,
            "input_dim": in_dim,
            "output_dim": out_dim,
            "rank_parameter_cost": in_dim + out_dim,
            "num_activation_tokens": num_tokens,
            "weighted_error_sq": weighted_error_sq,
            "weighted_output_sq": weighted_output_sq,
            "weighted_residual_singular_values": singular_values,
            "spectral_rank_computed": max_spectral_rank,
        })

        # Free gradient memory.
        module.weight.grad = None

    results.sort(key=lambda x: (x.get("layer_id") or 0, x["projection"]))
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Gradient-based module importance for AdaLoRA rank allocation."
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config (adalora_qwen3_1p7b.yaml).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path.  Defaults to <model_output_dir>/gradient_importance.json.",
    )
    parser.add_argument(
        "--max_calibration_samples", type=int, default=64,
        help="Number of calibration examples to use.",
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=512,
        help="Maximum token length per calibration example.",
    )
    parser.add_argument(
        "--max_spectral_rank", type=int, default=64,
        help="Number of top singular values to compute per module.",
    )
    parser.add_argument(
        "--svd_niter", type=int, default=4,
        help="Power iterations for randomized SVD.",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)

    # Resolve output path.
    if args.output is None:
        output_dir = config["output"]["model_output_dir"]
        output_path = os.path.join(output_dir, "gradient_importance.json")
    else:
        output_path = args.output
    ensure_dir(os.path.dirname(os.path.abspath(output_path)))

    # Disable quantization so the model runs in full precision.
    config.pop("quantization", None)
    # Cap calibration samples.
    config.setdefault("dataset", {})
    config["dataset"]["max_train_samples"] = args.max_calibration_samples

    print("=" * 70)
    print("Gradient importance analysis (AdaLoRA without quantization)")
    print(f"Model : {config['model']['model_name']}")
    print(f"Output: {output_path}")
    print("=" * 70)

    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(config)
    device = next(model.parameters()).device

    print("Loading calibration dataset...")
    dataset = load_gsm8k_for_sft(config)
    if len(dataset) > args.max_calibration_samples:
        dataset = dataset.select(range(args.max_calibration_samples))

    def _tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
        )

    tokenized = dataset.map(
        _tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing calibration data",
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataloader = DataLoader(
        tokenized,
        batch_size=1,
        collate_fn=collator,
        shuffle=False,
    )

    target_suffixes = config.get("lora", {}).get(
        "target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    target_modules = get_target_linear_modules(model, target_suffixes)
    print(f"Found {len(target_modules)} target modules.")

    print(f"Running {args.max_calibration_samples} calibration forward-backward passes...")
    results = compute_gradient_importance(
        model=model,
        target_modules=target_modules,
        dataloader=dataloader,
        max_calibration_steps=args.max_calibration_samples,
        max_spectral_rank=args.max_spectral_rank,
        svd_niter=args.svd_niter,
    )

    report = {
        "model_name": config["model"]["model_name"],
        "score_method": "gradient_svd",
        "calibration_samples": args.max_calibration_samples,
        "max_spectral_rank": args.max_spectral_rank,
        "num_modules": len(results),
        "modules": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved gradient importance to: {output_path}")
    print(f"Total modules: {len(results)}")


if __name__ == "__main__":
    main()
