from typing import Dict

from datasets import Dataset, load_dataset


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

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
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
    """
    Format GSM8K example for generation/evaluation.

    The model sees only the question and must generate the answer.
    """
    question = example["question"].strip()

    return {
        "question": question,
        "answer": example["answer"].strip(),
    }


def load_gsm8k_for_eval(config):
    """
    Load GSM8K test set for evaluation.
    """
    from datasets import load_dataset

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