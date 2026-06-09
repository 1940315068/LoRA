import re
from typing import Optional
import torch

def extract_final_number(text: str) -> Optional[str]:
    """
    Extract the final numeric answer from model output.

    This is a simple GSM8K-style extractor.
    """
    if text is None:
        return None

    # Prefer answer after #### if present
    if "####" in text:
        text = text.split("####")[-1]

    numbers = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))

    if not numbers:
        return None

    return numbers[-1]


def is_correct(prediction: str, gold: str) -> bool:
    """
    Compare predicted final number with gold final number.
    """
    pred_answer = extract_final_number(prediction)
    gold_answer = extract_final_number(gold)

    if pred_answer is None or gold_answer is None:
        return False

    return pred_answer == gold_answer


def reset_peak_gpu_memory():
    """
    Reset peak allocated memory stats for all visible GPUs.
    """
    if not torch.cuda.is_available():
        return

    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)

    torch.cuda.synchronize()


def get_peak_gpu_memory_gb():
    """
    Return peak allocated memory across all visible GPUs.
    """
    if not torch.cuda.is_available():
        return None

    peak_memory = 0.0

    for i in range(torch.cuda.device_count()):
        peak_memory = max(
            peak_memory,
            torch.cuda.max_memory_allocated(i) / 1024**3,
        )

    return peak_memory