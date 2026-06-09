# LoRA / QLoRA Rank Allocation Project

This project studies LoRA and QLoRA fine-tuning, and later explores quantization-error-guided LoRA rank allocation.

Current stage: **LoRA rank sweep on Qwen3-1.7B with GSM8K**.

## Project Structure

```text
src/
  train.py                 # train LoRA adapters
  evaluate.py              # evaluate base model or LoRA adapters
  summarize_results.py     # collect train/eval results into one CSV

  configs/
    lora_qwen3_1p7b.yaml   # shared config for LoRA experiments

  utils/
    dataset_utils.py       # load and format GSM8K
    model_utils.py         # load model and tokenizer
    lora_utils.py          # LoRA config and parameter counting
    eval_utils.py          # answer extraction, accuracy, GPU memory utils
    io_utils.py            # YAML/JSON helpers and config override

  outputs/
    qwen3_1p7b/
      base/                # base model evaluation
      lora_r4/             # LoRA r=4 results
      lora_r8/             # LoRA r=8 results
      lora_r16/            # LoRA r=16 results
      ...
```

## Setup

```bash
conda create -n lora python=3.11 -y
conda activate lora
pip install -r requirements.txt
```

## Train LoRA

Use the shared config and override the LoRA rank from the command line.

```bash
python -m src.train --config src/configs/lora_qwen3_1p7b.yaml --rank 16
```

The adapter and training metrics will be saved to:

```text
src/outputs/qwen3_1p7b/lora_r16/
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

## Evaluate LoRA Adapter

```bash
python -m src.evaluate \
  --config src/configs/lora_qwen3_1p7b.yaml \
  --rank 16 \
  --adapter src/outputs/qwen3_1p7b/lora_r16/adapter
```

Results are saved to:

```text
src/outputs/qwen3_1p7b/lora_r16/lora_eval_results.json
src/outputs/qwen3_1p7b/lora_r16/lora_predictions.jsonl
```

## LoRA Rank Sweep

Example ranks:

```bash
python -m src.train --config src/configs/lora_qwen3_1p7b.yaml --rank 4
python -m src.evaluate --config src/configs/lora_qwen3_1p7b.yaml --rank 4 --adapter src/outputs/qwen3_1p7b/lora_r4/adapter

python -m src.train --config src/configs/lora_qwen3_1p7b.yaml --rank 8
python -m src.evaluate --config src/configs/lora_qwen3_1p7b.yaml --rank 8 --adapter src/outputs/qwen3_1p7b/lora_r8/adapter

python -m src.train --config src/configs/lora_qwen3_1p7b.yaml --rank 16
python -m src.evaluate --config src/configs/lora_qwen3_1p7b.yaml --rank 16 --adapter src/outputs/qwen3_1p7b/lora_r16/adapter

python -m src.train --config src/configs/lora_qwen3_1p7b.yaml --rank 32
python -m src.evaluate --config src/configs/lora_qwen3_1p7b.yaml --rank 32 --adapter src/outputs/qwen3_1p7b/lora_r32/adapter

python -m src.train --config src/configs/lora_qwen3_1p7b.yaml --rank 64
python -m src.evaluate --config src/configs/lora_qwen3_1p7b.yaml --rank 64 --adapter src/outputs/qwen3_1p7b/lora_r64/adapter
```

## Summarize Results

```bash
python -m src.summarize_results
```

This creates:

```text
src/outputs/qwen3_1p7b/summary.csv
```

To summarize only selected ranks:

```bash
python -m src.summarize_results --ranks 16 64
```