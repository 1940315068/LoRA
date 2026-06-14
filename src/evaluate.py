import argparse
import json
import os
import time

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.dataset_utils import load_dataset_for_eval
from src.utils.io_utils import (
    ensure_dir,
    load_yaml,
    save_yaml,
    save_json,
    apply_model_override,
    apply_method_override,
    apply_lora_rank_override,
    apply_output_dir,
    infer_dataset_task,
    get_base_eval_output_dir,
    get_experiment_output_dir,
)
from src.utils.model_utils import get_torch_dtype, load_base_model, load_tokenizer
from src.utils.eval_utils import (
    is_correct,
    is_correct_math,
    is_correct_optmath,
    extract_final_number,
    extract_math_final_answer,
    extract_python_code,
    run_python_code,
    extract_optimal_value,
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
    """
    if adapter_path is None:
        output_dir = get_base_eval_output_dir(config)
        eval_output_path = os.path.join(output_dir, "eval_results.json")
        pred_output_path = os.path.join(output_dir, "predictions.jsonl")
        model_type = "base"

    else:
        method = config.get("experiment", {}).get("method", "lora")
        output_dir = get_experiment_output_dir(config)
        eval_output_path = os.path.join(output_dir, f"{method}_eval_results.json")
        pred_output_path = os.path.join(output_dir, f"{method}_predictions.jsonl")
        model_type = method

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


def build_qwen3_eval_prompt(tokenizer, question: str, dataset_name: str = "gsm8k") -> str:
    if dataset_name == "optmath":
        instruction = (
            "Write executable Python code using gurobipy to formulate and solve "
            "the following optimization problem. "
            "At the end, print the optimal objective value exactly in this format:\n"
            "OPTIMAL_VALUE: <value>\n\n"
        )
    elif dataset_name == "math":
        instruction = (
            "Solve the following competition math problem step by step. "
            "The final answer may be a number, expression, interval, set, or LaTeX expression. "
            "Put only the final answer after ####.\n\n"
        )
    else:
        instruction = (
            "Solve the following math problem step by step. "
            "Put the final numerical answer after ####.\n\n"
        )

    messages = [
        {
            "role": "user",
            "content": instruction + question,
        }
    ]

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        

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
    parser.add_argument(
        "--model_key",
        type=str,
        default=None,
        help="Model key, e.g., qwen3_1p7b, qwen3_4b, qwen3_8b.",
    )

    parser.add_argument(
        "--method",
        type=str,
        default=None,
        choices=["lora", "qlora"],
        help="Evaluation method.",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    config = apply_model_override(config, model_key=args.model_key)
    config = apply_method_override(config, method=args.method)
    config = apply_lora_rank_override(config, rank=args.rank)
    config = apply_output_dir(config)
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

    dataset_cfg = config.get("dataset", {})
    task_name = infer_dataset_task(config)
    dataset_name = dataset_cfg.get("dataset_name", "unknown")
    dataset_config = dataset_cfg.get("dataset_config", None)
    eval_split = config.get("evaluation", {}).get("split", "test")

    print(
        f"Loading {task_name} eval set: "
        f"name={dataset_name}, config={dataset_config}, split={eval_split}"
    )

    dataset = load_dataset_for_eval(config)

    print(f"Loaded eval examples: {len(dataset)}")

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
    code_generated_count = 0
    code_executed_count = 0

    reset_peak_gpu_memory()

    start_time = time.time()

    print("Generating answers...")
    for example in tqdm(dataset):
        prompt = build_qwen3_eval_prompt(
            tokenizer=tokenizer,
            question=example["question"],
            dataset_name=task_name,
        )
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

        if task_name in ["math", "hendrycks_math"]:
            ok = is_correct_math(pred, gold)
            gold_final = extract_math_final_answer(gold)
            pred_final = extract_math_final_answer(pred)

            code = None
            run_result = None
            code_generated = None
            code_executed = None
            judge_reason = "math_text_match"

        elif task_name == "optmath":
            code = extract_python_code(pred)
            code_generated = code is not None

            if code_generated:
                timeout = eval_cfg.get("execution_timeout", 100)
                run_result = run_python_code(code, timeout=timeout)
                code_executed = run_result["executed"]
                pred_final = extract_optimal_value(run_result["stdout"])
            else:
                run_result = {
                    "executed": False,
                    "timeout": False,
                    "returncode": None,
                    "stdout": "",
                    "stderr": "No Python code generated.",
                }
                code_executed = False
                pred_final = None

            gold_final = gold
            tolerance = eval_cfg.get("numerical_tolerance", 0.05)
            ok = is_correct_optmath(pred_final, gold_final, tolerance=tolerance)

            if ok:
                judge_reason = "correct"
            elif not code_generated:
                judge_reason = "no_code_generated"
            elif not code_executed:
                judge_reason = "execution_failed"
            elif pred_final is None:
                judge_reason = "no_objective_value"
            else:
                judge_reason = "wrong_objective_value"

        elif task_name == "gsm8k":
            ok = is_correct(pred, gold)
            gold_final = extract_final_number(gold)
            pred_final = extract_final_number(pred)

            code = None
            run_result = None
            code_generated = None
            code_executed = None
            judge_reason = "numeric_match"

        else:
            raise ValueError(f"Unsupported eval task: {task_name}")

        total += 1
        correct += int(ok)

        if task_name == "optmath":
            code_generated_count += int(code_generated)
            code_executed_count += int(code_executed)

        item = {
            "question": example["question"],
            "prompt": prompt,
            "gold_answer": gold,
            "prediction": pred,
            "gold_final": gold_final,
            "pred_final": pred_final,
            "correct": ok,
        }

        if task_name == "optmath":
            item.update(
                {
                    "generated_code": code,
                    "code_generated": code_generated,
                    "code_executed": code_executed,
                    "execution_stdout": run_result["stdout"],
                    "execution_stderr": run_result["stderr"],
                    "execution_returncode": run_result["returncode"],
                    "execution_timeout": run_result["timeout"],
                    "judge_reason": judge_reason,
                }
            )

        predictions.append(item)

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

    if task_name == "optmath":
        results.update(
            {
                "code_generated": code_generated_count,
                "code_executed": code_executed_count,
                "generation_success_rate": code_generated_count / total if total > 0 else 0.0,
                "execution_success_rate": code_executed_count / total if total > 0 else 0.0,
                "numerical_tolerance": eval_cfg.get("numerical_tolerance", 0.05),
                "execution_timeout": eval_cfg.get("execution_timeout", 100),
            }
        )

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