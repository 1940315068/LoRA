# LoRA / QLoRA / AdaLoRA Rank Allocation Project

This project studies LoRA, QLoRA, and AdaLoRA fine-tuning, and explores quantization-error-guided and gradient-guided LoRA rank allocation.

Current stage: **LoRA / QLoRA / AdaLoRA rank sweep on Qwen3-1.7B with GSM8K**.

## Project Structure

```text
src/
  train.py                          # train LoRA / QLoRA / AdaLoRA adapters
  evaluate.py                       # evaluate base model or trained adapters
  run_rank_sweep.py                 # run rank sweep across methods
  summarize_results.py              # collect train/eval results into one CSV
  compute_effective_rank.py         # compute effective rank of trained adapters
  summarize_effective_rank.py       # collect effective rank results into one CSV
  compute_gradient_importance.py    # (AdaLoRA) compute per-module gradient importance
  create_adalora_rank_config.py     # (AdaLoRA) greedy per-module rank allocation

  configs/
    lora_qwen3_1p7b.yaml            # shared config for LoRA experiments
    qlora_qwen3_1p7b.yaml           # shared config for QLoRA experiments
    adalora_qwen3_1p7b.yaml         # shared config for AdaLoRA experiments

  utils/
    dataset_utils.py                # load and format GSM8K
    model_utils.py                  # load full-precision or 4-bit model
    lora_utils.py                   # LoRA / QLoRA setup and parameter counting
    eval_utils.py                   # answer extraction, accuracy, GPU memory utils
    io_utils.py                     # YAML/JSON helpers and config override
    rank_analysis_utils.py          # effective rank computation utilities

  outputs/
    qwen3_1p7b/
      base/                         # base model evaluation
      lora_r4/                      # LoRA r=4 results
      lora_r8/ ... lora_r64/
      qlora_r4/                     # QLoRA r=4 results
      qlora_r8/ ... qlora_r64/
      adalora_avg4_output_normalized/   # AdaLoRA avg_rank=4 results
      adalora_avg8_output_normalized/
      ... adalora_avg64_output_normalized/
      rank_configs/                 # per-rank AdaLoRA allocation JSONs
      gradient_importance.json      # pre-computed gradient importance (shared)
      summary.csv
      effective_rank_summary.csv
```

## Setup

```bash
conda create -n lora python=3.11 -y
conda activate lora
pip install -r requirements.txt
```

## Train LoRA / QLoRA

Use the shared config and override the rank from the command line.

LoRA:

```bash
python -m src.train --config src/configs/lora_qwen3_1p7b.yaml --rank 16
```

QLoRA:

```bash
python -m src.train --config src/configs/qlora_qwen3_1p7b.yaml --rank 16
```

The adapter and training metrics will be saved to:

```text
src/outputs/qwen3_1p7b/lora_r16/
src/outputs/qwen3_1p7b/qlora_r16/
```

## Train AdaLoRA

AdaLoRA uses **static per-module rank allocation** guided by gradient importance scores.
The pipeline has three steps before training.

**Step 1 — Compute gradient importance (once, shared across all ranks):**

```bash
python -m src.compute_gradient_importance \
    --config src/configs/adalora_qwen3_1p7b.yaml \
    --output src/outputs/qwen3_1p7b/gradient_importance.json
```

This runs a short calibration pass (64 samples) and performs SVD on the accumulated
gradients for each target module, producing an importance score per module.

**Step 2 — Generate per-module rank allocation (once per target avg_rank):**

```bash
python -m src.create_adalora_rank_config \
    --importance src/outputs/qwen3_1p7b/gradient_importance.json \
    --output     src/outputs/qwen3_1p7b/rank_configs/adalora_rank_avg16.json \
    --avg_rank   16 --min_rank 2 --max_rank 256 --rank_step 2 \
    --score_mode output_normalized
```

This uses a greedy algorithm to allocate ranks across all target modules so that
the total parameter budget equals `avg_rank × num_modules`, with important modules
getting higher ranks.

**Step 3 — Train with the rank allocation:**

```bash
python -m src.train \
    --config               src/configs/adalora_qwen3_1p7b.yaml \
    --adaptive_rank_config src/outputs/qwen3_1p7b/rank_configs/adalora_rank_avg16.json
```

The experiment name and output directory are derived automatically from the rank config:

```text
src/outputs/qwen3_1p7b/adalora_avg16_output_normalized/
```

## Evaluate Base Model

The base model only needs to be evaluated once.

```bash
python -m src.evaluate --config src/configs/lora_qwen3_1p7b.yaml
```

Results are saved to:

```text
src/outputs/qwen3_1p7b/base/
```

If the result already exists, evaluation is skipped. To rerun:

```bash
python -m src.evaluate --config src/configs/lora_qwen3_1p7b.yaml --overwrite
```

## Evaluate LoRA / QLoRA Adapter

If `--rank` is provided, the adapter path is inferred automatically.

LoRA:

```bash
python -m src.evaluate --config src/configs/lora_qwen3_1p7b.yaml --rank 16
```

QLoRA:

```bash
python -m src.evaluate --config src/configs/qlora_qwen3_1p7b.yaml --rank 16
```

Results are saved to:

```text
src/outputs/qwen3_1p7b/lora_r16/lora_eval_results.json
src/outputs/qwen3_1p7b/qlora_r16/lora_eval_results.json
```

## Evaluate AdaLoRA Adapter

Pass `--adapter` with the path to the adapter directory directly:

```bash
python -m src.evaluate \
    --config  src/configs/adalora_qwen3_1p7b.yaml \
    --adapter src/outputs/qwen3_1p7b/adalora_avg16_output_normalized/adapter
```

Results are saved next to the adapter:

```text
src/outputs/qwen3_1p7b/adalora_avg16_output_normalized/lora_eval_results.json
```

## Rank Sweep

Example for LoRA:

```bash
python -m src.train --config src/configs/lora_qwen3_1p7b.yaml --rank 4
python -m src.evaluate --config src/configs/lora_qwen3_1p7b.yaml --rank 4
```

Example for QLoRA:

```bash
python -m src.train --config src/configs/qlora_qwen3_1p7b.yaml --rank 4
python -m src.evaluate --config src/configs/qlora_qwen3_1p7b.yaml --rank 4
```

Example for AdaLoRA (steps 1–2 only need to run once and once-per-rank respectively):

```bash
# Step 1: compute gradient importance (once)
python -m src.compute_gradient_importance \
    --config src/configs/adalora_qwen3_1p7b.yaml \
    --output src/outputs/qwen3_1p7b/gradient_importance.json

# Step 2+3: for each avg_rank
python -m src.create_adalora_rank_config \
    --importance src/outputs/qwen3_1p7b/gradient_importance.json \
    --output src/outputs/qwen3_1p7b/rank_configs/adalora_rank_avg4.json \
    --avg_rank 4 --min_rank 2 --max_rank 256 --rank_step 2 --score_mode output_normalized
python -m src.train --config src/configs/adalora_qwen3_1p7b.yaml \
    --adaptive_rank_config src/outputs/qwen3_1p7b/rank_configs/adalora_rank_avg4.json
python -m src.evaluate --config src/configs/adalora_qwen3_1p7b.yaml \
    --adapter src/outputs/qwen3_1p7b/adalora_avg4_output_normalized/adapter
```

Repeat with ranks:

```text
4, 8, 16, 32, 64
```

Or run all methods and ranks in one command:

```bash
python -m src.run_rank_sweep --methods lora qlora adalora --ranks 4 8 16 32 64
```

The sweep script handles the full AdaLoRA pipeline automatically (gradient importance
computed once; rank configs and training repeated per rank).

## Summarize Results

```bash
python -m src.summarize_results
```
By default, this summarizes LoRA, QLoRA, and AdaLoRA results.

This creates:

```text
src/outputs/qwen3_1p7b/summary.csv
```

To summarize only selected ranks:

```bash
python -m src.summarize_results --ranks 16 64
```

To summarize only selected methods:

```bash
python -m src.summarize_results --methods lora
python -m src.summarize_results --methods qlora
python -m src.summarize_results --methods adalora
```

To summarize selected methods and ranks:

```bash
python -m src.summarize_results --methods lora qlora adalora --ranks 16 64
```

## Effective Rank Analysis

Compute effective ranks of trained LoRA / QLoRA adapters:

```bash
python -m src.compute_effective_rank --adapter src/outputs/qwen3_1p7b/lora_r64/adapter
python -m src.compute_effective_rank --adapter src/outputs/qwen3_1p7b/qlora_r64/adapter
```

For AdaLoRA adapters, pass the adapter path directly:

```bash
python -m src.compute_effective_rank --adapter src/outputs/qwen3_1p7b/adalora_avg16_output_normalized/adapter
```

Summarize all effective rank results:

```bash
python -m src.summarize_effective_rank
```

This creates:

```text
src/outputs/qwen3_1p7b/effective_rank_summary.csv
```