# LoRA / QLoRA Rank Allocation Project

This project studies LoRA and QLoRA fine-tuning, and later explores quantization-error-guided LoRA rank allocation.

Current stage: **LoRA and QLoRA rank sweep on Qwen3-1.7B with GSM8K**.

## Project Structure

```text
src/
  train.py                  # train LoRA / QLoRA adapters
  evaluate.py               # evaluate base model or trained adapters
  summarize_results.py      # collect train/eval results into one CSV

  configs/
    lora_qwen3_1p7b.yaml    # shared config for LoRA experiments
    qlora_qwen3_1p7b.yaml   # shared config for QLoRA experiments

  utils/
    dataset_utils.py        # load and format GSM8K
    model_utils.py          # load full-precision or 4-bit model
    lora_utils.py           # LoRA / QLoRA setup and parameter counting
    eval_utils.py           # answer extraction, accuracy, GPU memory utils
    io_utils.py             # YAML/JSON helpers and config override

  outputs/
    qwen3_1p7b/
      base/                 # base model evaluation
      lora_r4/              # LoRA r=4 results
      lora_r8/
      ...
      qlora_r4/             # QLoRA r=4 results
      qlora_r8/
      ...
      summary.csv
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

Repeat with ranks:

```text
4, 8, 16, 32, 64
```

## Summarize Results

```bash
python -m src.summarize_results
```
By default, this summarizes both LoRA and QLoRA results.

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
```

To summarize selected methods and ranks:

```bash
python -m src.summarize_results --methods lora qlora --ranks 16 64
```