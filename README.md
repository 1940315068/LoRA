# LoRA / QLoRA Rank Allocation Project

This project studies LoRA / QLoRA fine-tuning and quantization-error-guided LoRA rank allocation.

## Project Structure

```text
.
├── README.md
├── requirements.txt
├── documents/
└── src/
    ├── train.py
    ├── evaluate.py
    ├── summarize_results.py
    ├── compute_effective_rank.py
    ├── summarize_rank.py
    ├── run_exps.sh
    ├── run_rank_analysis.sh
    ├── configs/
    │   ├── qwen3.yaml
    │   ├── qwen3_math.yaml
    │   └── qwen3_optmath.yaml
    ├── outputs/
    └── utils/
        ├── dataset_utils.py
        ├── model_utils.py
        ├── lora_utils.py
        ├── eval_utils.py
        ├── io_utils.py
        └── rank_analysis_utils.py
```

Config files:

```text
src/configs/qwen3.yaml
src/configs/qwen3_math.yaml
src/configs/qwen3_optmath.yaml
```

Use `--config` to choose which config file to run.


## Setup

Create a Python 3.11 environment and install dependencies:

```bash
conda create -n lora python=3.11
conda activate lora

pip install -r requirements.txt
```

## Train

Example:

```bash
python -m src.train \
  --config src/configs/qwen3_math.yaml \
  --model_key qwen3_8b \
  --method lora \
  --rank 16 \
  --learning_rate 1e-4
```

Training options:

```text
--config          Path to config yaml.
--model_key       Model key, e.g. qwen3_1p7b, qwen3_4b, qwen3_8b.
--method          Fine-tuning method: lora or qlora.
--rank            LoRA/QLoRA rank.
--learning_rate   Learning rate for training.
```

Supported model keys:

```text
qwen3_1p7b qwen3_4b qwen3_8b
```

Supported methods:

```text
lora qlora
```

Outputs are saved under:

```text
src/outputs/<model_key>/<task_name>/<method>_r<rank>/
```

For example:

```text
src/outputs/qwen3_8b/optmath/lora_r16/
```

## Evaluate

Evaluate a trained adapter:

```bash
python -m src.evaluate \
  --config src/configs/qwen3_math.yaml \
  --model_key qwen3_8b \
  --method lora \
  --rank 16
```

Evaluate options:

```text
--config         Path to config yaml.
--model_key      Model key, e.g. qwen3_1p7b, qwen3_4b, qwen3_8b.
--method         Fine-tuning method: lora or qlora.
--rank           LoRA/QLoRA rank.
```

Evaluate the base model:

```bash
python -m src.evaluate \
  --config src/configs/qwen3_math.yaml \
  --model_key qwen3_8b \
  --method lora
```

When evaluating the base model, no `--rank` is needed.


## Summarize Results

Summarize evaluation results under one model output directory:

```bash
python -m src.summarize_results \
  --output_root src/outputs/qwen3_8b/optmath
```

Optional filters:

```bash
python -m src.summarize_results \
  --output_root src/outputs/qwen3_8b/optmath \
  --methods qlora \
  --ranks 4 16 64
```

Options:

```text
--output_root  Output directory to summarize.
--methods      Optional method filter.
--ranks        Optional rank filter.
```

## Effective Rank Analysis

Compute effective rank for one adapter:

```bash
python -m src.compute_effective_rank \
  --adapter src/outputs/qwen3_8b/optmath/qlora_r16/adapter
```

Summarize rank results:

```bash
python -m src.summarize_rank \
  --output_root src/outputs/qwen3_8b/optmath
```

This creates:

```text
src/outputs/qwen3_8b/optmath/effective_rank_summary.csv
```
