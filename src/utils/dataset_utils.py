from typing import Dict, List, Optional
from datasets import Dataset, concatenate_datasets, load_dataset
from src.utils.io_utils import infer_dataset_task


MATH_SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


def apply_chat_template(tokenizer, messages, add_generation_prompt: bool = False) -> str:
    """
    Qwen3 supports enable_thinking=False, but some tokenizers may not.
    This helper keeps the code robust.
    """
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


# ============================================================
# GSM8K
# ============================================================

def format_gsm8k_example(example: Dict[str, str], tokenizer) -> Dict[str, str]:
    question = example["question"].strip()
    answer = example["answer"].strip()

    messages = [
        {
            "role": "user",
            "content": (
                "Solve the following math problem step by step. "
                "Put the final numerical answer after ####.\n\n"
                f"{question}"
            ),
        },
        {
            "role": "assistant",
            "content": answer,
        },
    ]

    text = apply_chat_template(
        tokenizer=tokenizer,
        messages=messages,
        add_generation_prompt=False,
    )

    return {"text": text}


def load_gsm8k_for_sft(config: Dict, tokenizer) -> Dataset:
    dataset_name = config["dataset"]["dataset_name"]
    dataset_config = config["dataset"].get("dataset_config", "main")
    split = config["dataset"].get("split", "train")
    max_train_samples = config["dataset"].get("max_train_samples", None)

    dataset = load_dataset(dataset_name, dataset_config, split=split)

    if max_train_samples is not None:
        max_train_samples = min(max_train_samples, len(dataset))
        dataset = dataset.select(range(max_train_samples))

    dataset = dataset.map(
        lambda example: format_gsm8k_example(example, tokenizer),
        remove_columns=dataset.column_names,
        desc="Formatting GSM8K SFT examples",
    )

    return dataset


def format_gsm8k_prompt(example):
    question = example["question"].strip()

    return {
        "question": question,
        "answer": example["answer"].strip(),
    }


def load_gsm8k_for_eval(config):
    dataset_name = config["dataset"]["dataset_name"]
    dataset_config = config["dataset"].get("dataset_config", "main")

    eval_split = config.get("evaluation", {}).get("split", "test")
    max_eval_samples = config.get("evaluation", {}).get("max_eval_samples", 100)

    dataset = load_dataset(dataset_name, dataset_config, split=eval_split)

    if max_eval_samples is not None:
        max_eval_samples = min(max_eval_samples, len(dataset))
        dataset = dataset.select(range(max_eval_samples))

    dataset = dataset.map(
        format_gsm8k_prompt,
        remove_columns=dataset.column_names,
        desc="Formatting GSM8K eval examples",
    )

    return dataset


# ============================================================
# MATH / Hendrycks MATH
# ============================================================

def extract_boxed_answer(text: str) -> Optional[str]:
    """
    Extract the last \\boxed{...} answer from a MATH solution.

    Handles nested braces, e.g.
        \\boxed{\\frac{3}{4}}
    """
    if text is None:
        return None

    marker = r"\boxed{"
    start = text.rfind(marker)

    if start == -1:
        return None

    i = start + len(marker)
    depth = 1
    j = i

    while j < len(text):
        ch = text[j]

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[i:j].strip()

        j += 1

    return None


def get_math_subjects(config: Dict) -> List[str]:
    """
    Read MATH subject config.

    Supported examples:
        dataset_config: all
        dataset_config: algebra
        dataset_config:
          - algebra
          - geometry
    """
    dataset_cfg = config.get("dataset", {})
    dataset_config = dataset_cfg.get("dataset_config", "all")

    if dataset_config == "all":
        return MATH_SUBJECTS

    if isinstance(dataset_config, str):
        return [dataset_config]

    if isinstance(dataset_config, list):
        return dataset_config

    raise ValueError(f"Invalid MATH dataset_config: {dataset_config}")


def load_math_raw(config: Dict, split: str) -> Dataset:
    """
    Load Hendrycks MATH.

    For EleutherAI/hendrycks_math, each subject is a separate config.
    If dataset_config is 'all', we concatenate all subjects.
    """
    dataset_name = config["dataset"]["dataset_name"]
    subjects = get_math_subjects(config)

    datasets = [
        load_dataset(dataset_name, subject, split=split)
        for subject in subjects
    ]

    if len(datasets) == 1:
        return datasets[0]

    return concatenate_datasets(datasets)


def format_math_example(example: Dict[str, str], tokenizer) -> Dict[str, str]:
    """
    Format MATH example for SFT.

    MATH has:
        problem: question
        solution: full worked solution

    We train the model to output the full solution and append:
        #### final_answer

    This makes evaluation easier and keeps the output format consistent
    with GSM8K.
    """
    question = example["problem"].strip()
    solution = example["solution"].strip()

    final_answer = extract_boxed_answer(solution)

    if final_answer is None:
        # Fallback: use the whole solution if no boxed answer is found.
        # This is not ideal, but prevents crashes.
        final_answer = solution

    assistant_content = solution

    if "####" not in assistant_content:
        assistant_content = assistant_content.rstrip() + f"\n\n#### {final_answer}"

    messages = [
        {
            "role": "user",
            "content": (
                "Solve the following competition math problem step by step. "
                "The final answer may be a number, expression, interval, set, "
                "or LaTeX expression. Put only the final answer after ####.\n\n"
                f"{question}"
            ),
        },
        {
            "role": "assistant",
            "content": assistant_content,
        },
    ]

    text = apply_chat_template(
        tokenizer=tokenizer,
        messages=messages,
        add_generation_prompt=False,
    )

    return {
        "text": text,
        "question": question,
        "answer": final_answer,
    }


def load_math_for_sft(config: Dict, tokenizer) -> Dataset:
    split = config["dataset"].get("split", "train")
    max_train_samples = config["dataset"].get("max_train_samples", None)

    dataset = load_math_raw(config, split=split)

    if max_train_samples is not None:
        max_train_samples = min(max_train_samples, len(dataset))
        dataset = dataset.select(range(max_train_samples))

    dataset = dataset.map(
        lambda example: format_math_example(example, tokenizer),
        remove_columns=dataset.column_names,
        desc="Formatting MATH SFT examples",
    )

    return dataset


def format_math_prompt(example: Dict[str, str]) -> Dict[str, str]:
    """
    Format MATH example for generation/evaluation.

    The model sees only the problem.
    The gold answer is extracted from the boxed answer in the solution.
    """
    question = example["problem"].strip()
    solution = example["solution"].strip()

    final_answer = extract_boxed_answer(solution)

    if final_answer is None:
        final_answer = solution

    return {
        "question": question,
        "answer": final_answer,
        "solution": solution,
        "level": example.get("level", None),
        "type": example.get("type", None),
    }


def load_math_for_eval(config: Dict) -> Dataset:
    eval_split = config.get("evaluation", {}).get("split", "test")
    max_eval_samples = config.get("evaluation", {}).get("max_eval_samples", 100)

    dataset = load_math_raw(config, split=eval_split)

    if max_eval_samples is not None:
        max_eval_samples = min(max_eval_samples, len(dataset))
        dataset = dataset.select(range(max_eval_samples))

    dataset = dataset.map(
        format_math_prompt,
        remove_columns=dataset.column_names,
        desc="Formatting MATH eval examples",
    )

    return dataset


# ============================================================
# OptMATH
# ============================================================

def get_optmath_question(example: Dict) -> str:
    """
    Robustly get OptMATH question text from different possible schemas.
    """
    question = (
        example.get("en_question")
        or example.get("question")
        or example.get("input")
        or ""
    )
    return str(question).strip()


def get_optmath_answer(example: Dict) -> str:
    """
    Robustly get OptMATH gold answer from different possible schemas.
    """
    answer = (
        example.get("en_answer")
        or example.get("answer")
        or ""
    )
    return str(answer).strip()


def format_optmath_prompt(example: Dict) -> Dict[str, str]:
    """
    Format OptMATH example for evaluation.
    Output schema is aligned with GSM8K/MATH:
        question
        answer
    """
    return {
        "question": get_optmath_question(example),
        "answer": get_optmath_answer(example),
    }


def load_optmath_for_eval(config: Dict) -> Dataset:
    dataset_name = config["dataset"]["dataset_name"]
    dataset_config = config["dataset"].get("dataset_config", None)

    eval_split = config.get("evaluation", {}).get("split", "test")
    max_eval_samples = config.get("evaluation", {}).get("max_eval_samples", 100)

    if dataset_config is None:
        dataset = load_dataset(dataset_name, split=eval_split)
    else:
        dataset = load_dataset(dataset_name, dataset_config, split=eval_split)

    if max_eval_samples is not None:
        max_eval_samples = min(max_eval_samples, len(dataset))
        dataset = dataset.select(range(max_eval_samples))

    dataset = dataset.map(
        format_optmath_prompt,
        remove_columns=dataset.column_names,
        desc="Formatting OptMATH eval examples",
    )

    return dataset


def format_optmath_example(example: Dict, tokenizer) -> Dict[str, str]:
    """
    Format OptMATH example for SFT.

    Expected common schema:
        instruction
        input
        output

    Fallback:
        question / en_question
        answer / en_answer
    """
    instruction = str(
        example.get("instruction")
        or "Build and solve the optimization model for the following problem using gurobipy."
    ).strip()

    question = get_optmath_question(example)

    assistant_content = str(
        example.get("output")
        or example.get("response")
        or example.get("solution")
        or get_optmath_answer(example)
    ).strip()

    user_content = instruction

    if question:
        user_content += "\n\n" + question

    messages = [
        {
            "role": "user",
            "content": user_content,
        },
        {
            "role": "assistant",
            "content": assistant_content,
        },
    ]

    text = apply_chat_template(
        tokenizer=tokenizer,
        messages=messages,
        add_generation_prompt=False,
    )

    return {"text": text}


def load_optmath_for_sft(config: Dict, tokenizer) -> Dataset:
    dataset_name = config["dataset"]["dataset_name"]
    dataset_config = config["dataset"].get("dataset_config", None)
    split = config["dataset"].get("split", "train")
    max_train_samples = config["dataset"].get("max_train_samples", None)

    if dataset_config is None:
        dataset = load_dataset(dataset_name, split=split)
    else:
        dataset = load_dataset(dataset_name, dataset_config, split=split)

    if max_train_samples is not None:
        max_train_samples = min(max_train_samples, len(dataset))
        dataset = dataset.select(range(max_train_samples))

    dataset = dataset.map(
        lambda example: format_optmath_example(example, tokenizer),
        remove_columns=dataset.column_names,
        desc="Formatting OptMATH SFT examples",
    )

    return dataset


# ============================================================
# Generic dispatchers
# ============================================================

def load_dataset_for_sft(config: Dict, tokenizer) -> Dataset:
    task = infer_dataset_task(config)

    if task == "optmath":
        return load_optmath_for_sft(config, tokenizer)

    if task == "gsm8k":
        return load_gsm8k_for_sft(config, tokenizer)

    if task in ["math", "hendrycks_math"]:
        return load_math_for_sft(config, tokenizer)

    raise ValueError(f"Unsupported SFT dataset task: {task}")


def load_dataset_for_eval(config: Dict) -> Dataset:
    task = infer_dataset_task(config)

    if task == "optmath":
        return load_optmath_for_eval(config)

    if task == "gsm8k":
        return load_gsm8k_for_eval(config)

    if task in ["math", "hendrycks_math"]:
        return load_math_for_eval(config)

    raise ValueError(f"Unsupported eval dataset task: {task}")