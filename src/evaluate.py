import argparse
import json
import os
import time

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.dataset_utils import load_gsm8k_for_eval
from src.utils.io_utils import ensure_dir, load_yaml, save_json, apply_lora_rank_override
from src.utils.model_utils import get_torch_dtype, load_base_model, load_tokenizer
from src.utils.eval_utils import (
    is_correct,
    extract_final_number,
    reset_peak_gpu_memory,
    get_peak_gpu_memory_gb,
)

def get_model_device(model):
    """
    Return the device of the first model parameter.
    This is safer than directly using model.device.
    """
    return next(model.parameters()).device


def load_model_for_eval(config, adapter_path: str = None):
    tokenizer = load_tokenizer(config)
    tokenizer.padding_side = "left"

    base_model = load_base_model(config)

    if adapter_path is not None:
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        model = base_model

    model.eval()

    return model, tokenizer


def get_eval_output_paths(config, adapter_path: str = None):
    """
    Decide where to save evaluation results.

    Base model results are shared under:
        src/outputs/qwen3_1p7b/base/

    LoRA results are saved under the current experiment output dir:
        src/outputs/qwen3_1p7b/lora_rxx/
    """
    if adapter_path is None:
        model_output_dir = config.get("output", {}).get("model_output_dir")

        if model_output_dir is None:
            # fallback: parent dir of training output
            model_output_dir = os.path.dirname(config["training"]["output_dir"])

        output_dir = os.path.join(model_output_dir, "base")
        eval_output_path = os.path.join(output_dir, "eval_results.json")
        pred_output_path = os.path.join(output_dir, "predictions.jsonl")
        model_type = "base"

    else:
        output_dir = config["training"]["output_dir"]
        eval_output_path = os.path.join(output_dir, "lora_eval_results.json")
        pred_output_path = os.path.join(output_dir, "lora_predictions.jsonl")
        model_type = "lora"

    return model_type, output_dir, eval_output_path, pred_output_path


@torch.no_grad()
def generate_answer(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    max_input_length: int = 1024,
):
    """
    Generate answer for one prompt.
    """
    device = get_model_device(model)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=max_input_length,
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if do_sample:
        generation_kwargs["temperature"] = temperature
    else:
        # Avoid warnings from Qwen generation config:
        # temperature/top_p/top_k are ignored when do_sample=False.
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None

    output_ids = model.generate(
        **inputs,
        **generation_kwargs,
    )

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return output_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Path to LoRA adapter directory. If not provided, evaluate the base model.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing evaluation results.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="Override LoRA rank r.",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    config = apply_lora_rank_override(config, rank=args.rank)   
    if args.adapter is None and args.rank is not None:
        args.adapter = os.path.join(
            config["training"]["output_dir"],
            "adapter",
        )
    if args.adapter is None:
        config["experiment_name"] = f"{config['model']['model_short_name']}_base"

    model_type, output_dir, eval_output_path, pred_output_path = get_eval_output_paths(
        config,
        args.adapter,
    )

    ensure_dir(output_dir)

    print("=" * 80)
    print(f"Evaluating experiment: {config['experiment_name']}")
    print(f"Model type: {model_type}")
    print(f"Adapter path: {args.adapter}")
    print("=" * 80)

    if args.adapter is None and os.path.exists(eval_output_path) and not args.overwrite:
        print("=" * 80)
        print("Base evaluation already exists. Skipping.")
        print(f"Existing result: {eval_output_path}")
        print("Use --overwrite to rerun base evaluation.")
        print("=" * 80)
        return

    if args.adapter is not None:
        print("Loading base model with LoRA adapter...")
    else:
        print("Loading base model only...")

    model, tokenizer = load_model_for_eval(config, args.adapter)

    print("Loading GSM8K test set...")
    dataset = load_gsm8k_for_eval(config)

    eval_cfg = config.get("evaluation", {})
    max_new_tokens = eval_cfg.get("max_new_tokens", 256)
    do_sample = eval_cfg.get("do_sample", False)
    temperature = eval_cfg.get("temperature", 0.0)
    max_input_length = eval_cfg.get(
        "max_input_length",
        config["training"].get("max_seq_length", 1024),
    )

    total = 0
    correct = 0
    predictions = []

    reset_peak_gpu_memory()

    start_time = time.time()

    print("Generating answers...")
    for example in tqdm(dataset):
        prompt = example["prompt"]
        gold = example["answer"]

        pred = generate_answer(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            max_input_length=max_input_length,
        )

        ok = is_correct(pred, gold)

        total += 1
        correct += int(ok)

        predictions.append(
            {
                "question": example["question"],
                "prompt": prompt,
                "gold_answer": gold,
                "prediction": pred,
                "gold_final": extract_final_number(gold),
                "pred_final": extract_final_number(pred),
                "correct": ok,
            }
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    eval_runtime = time.time() - start_time

    peak_memory_gb = get_peak_gpu_memory_gb()

    accuracy = correct / total if total > 0 else 0.0
    eval_samples_per_second = total / eval_runtime if eval_runtime > 0 else None
    avg_generation_time_per_sample = eval_runtime / total if total > 0 else None

    results = {
        "experiment_name": config["experiment_name"],
        "model_type": model_type,
        "adapter": args.adapter,
        "num_eval_samples": total,
        "correct": correct,
        "accuracy": accuracy,
        "eval_runtime": eval_runtime,
        "eval_samples_per_second": eval_samples_per_second,
        "avg_generation_time_per_sample": avg_generation_time_per_sample,
        "peak_gpu_memory_gb": peak_memory_gb,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature,
    }

    save_json(results, eval_output_path)

    with open(pred_output_path, "w", encoding="utf-8") as f:
        for item in predictions:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("=" * 80)
    print("Evaluation finished.")
    print(f"Model type: {model_type}")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Eval runtime: {eval_runtime:.2f} seconds")
    print(f"Avg generation time/sample: {avg_generation_time_per_sample:.2f} seconds")
    print(f"Eval samples/second: {eval_samples_per_second:.4f}")

    if peak_memory_gb is not None:
        print(f"Peak GPU memory: {peak_memory_gb:.2f} GB")

    print(f"Results saved to: {eval_output_path}")
    print(f"Predictions saved to: {pred_output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()