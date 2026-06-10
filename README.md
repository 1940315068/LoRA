# LoRA / QLoRA Rank Allocation Project

This project studies LoRA and QLoRA fine-tuning, and later explores quantization-error-guided LoRA rank allocation.

## Project Structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
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
    │   └── qwen3.yaml
    ├── outputs/
    └── utils/
        ├── dataset_utils.py
        ├── model_utils.py
        ├── lora_utils.py
        ├── eval_utils.py
        ├── io_utils.py
        └── rank_analysis_utils.py
```

## Configuration

All Qwen3 experiments use a single shared config:

```text
src/configs/qwen3.yaml
```

The model, method, and rank are selected from the command line:

```bash
--model_key qwen3_1p7b
--method lora
--rank 16
```

Supported model keys:

```text
qwen3_1p7b qwen3_4b qwen3_8b
```

Supported methods:

```text
lora qlora
```

## Train and Evaluate

Example: train and evaluate Qwen3-8B with LoRA rank 16.

```bash
python -m src.train \
  --config src/configs/qwen3.yaml \
  --model_key qwen3_8b \
  --method lora \
  --rank 16

python -m src.evaluate \
  --config src/configs/qwen3.yaml \
  --model_key qwen3_8b \
  --method lora \
  --rank 16
```

Example: train and evaluate Qwen3-8B with QLoRA rank 16.

```bash
python -m src.train \
  --config src/configs/qwen3.yaml \
  --model_key qwen3_8b \
  --method qlora \
  --rank 16

python -m src.evaluate \
  --config src/configs/qwen3.yaml \
  --model_key qwen3_8b \
  --method qlora \
  --rank 16
```

Outputs are saved under:

```text
src/outputs/<model_key>/<method>_r<rank>/
```

For example:

```text
src/outputs/qwen3_8b/qlora_r16/
```

## Run Experiments in Batch

Run training and evaluation for multiple models, methods, and ranks:

```bash
bash src/run_exps.sh \
  --models qwen3_1p7b,qwen3_4b,qwen3_8b \
  --methods lora,qlora \
  --ranks 4,8,16,32,64
```

## Summarize Performance Results

```bash
python -m src.summarize_results \
  --output_root src/outputs/qwen3_1p7b
```

For Qwen3-4B:

```bash
python -m src.summarize_results \
  --output_root src/outputs/qwen3_4b
```

For Qwen3-8B:

```bash
python -m src.summarize_results \
  --output_root src/outputs/qwen3_8b
```

To summarize selected methods and ranks:

```bash
python -m src.summarize_results \
  --output_root src/outputs/qwen3_8b \
  --methods qlora \
  --ranks 4 16 64
```

## Effective Rank Analysis

Compute effective rank for one adapter:

```bash
python -m src.compute_effective_rank \
  --adapter src/outputs/qwen3_8b/qlora_r16/adapter
```

Run effective rank analysis in batch:

```bash
bash src/run_rank_analysis.sh \
  --models qwen3_1p7b qwen3_4b qwen3_8b \
  --methods lora qlora \
  --ranks 4 8 16 32 64
```

Summarize effective rank results:

```bash
python -m src.summarize_rank \
  --output_root src/outputs/qwen3_8b
```

This creates:

```text
src/outputs/qwen3_8b/effective_rank_summary.csv
```
