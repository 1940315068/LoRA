from typing import Dict

from datasets import Dataset, load_dataset


def format_gsm8k_example(example: Dict[str, str]) -> Dict[str, str]:
    """
    Convert a GSM8K example into a simple supervised fine-tuning text format.

    For the first experiment, we train on the full text:
        Question: ...
        Answer: ...
    """
    question = example["question"].strip()
    answer = example["answer"].strip()

    text = (
        "Solve the following math problem.\n\n"
        f"Question: {question}\n\n"
        f"Answer: {answer}"
    )

    return {"text": text}


def load_gsm8k_for_sft(config: Dict) -> Dataset:
    """
    Load GSM8K and return a dataset with a single 'text' field.
    """
    dataset_name = config["dataset"]["dataset_name"]
    dataset_config = config["dataset"].get("dataset_config", "main")
    split = config["dataset"].get("split", "train")
    max_train_samples = config["dataset"].get("max_train_samples", None)

    dataset = load_dataset(dataset_name, dataset_config, split=split)

    if max_train_samples is not None:
        max_train_samples = min(max_train_samples, len(dataset))
        dataset = dataset.select(range(max_train_samples))

    dataset = dataset.map(
        format_gsm8k_example,
        remove_columns=dataset.column_names,
        desc="Formatting GSM8K examples",
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