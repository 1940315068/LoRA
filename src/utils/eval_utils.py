import re
from typing import Optional
import torch


try:
    from math_verify import parse, verify
    HAS_MATH_VERIFY = True
except Exception:
    HAS_MATH_VERIFY = False


def extract_math_final_answer(text: str):
    """
    Extract final answer for MATH-style outputs.

    Priority:
    1. content after ####
    2. last \\boxed{...}
    3. last non-empty line
    """
    if text is None:
        return ""

    text = str(text).strip()

    # 1. Prefer #### format
    if "####" in text:
        return text.split("####")[-1].strip()

    # 2. Prefer last \boxed{...}
    boxed = extract_last_boxed(text)
    if boxed is not None:
        return boxed.strip()

    # 3. Fallback: last non-empty line
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        return lines[-1]

    return text


def extract_last_boxed(text: str):
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
                return text[i:j]

        j += 1

    return None


def normalize_math_answer(text: str):
    if text is None:
        return ""

    text = str(text).strip()

    # remove common wrappers
    text = text.replace("$", "")
    text = text.replace("\\left", "")
    text = text.replace("\\right", "")
    text = text.replace("\\,", "")
    text = text.replace("\\!", "")
    text = text.strip()

    # remove trailing punctuation
    text = text.rstrip(".。")

    # normalize spaces
    text = re.sub(r"\s+", "", text)

    return text


def maybe_wrap_latex(text: str):
    """
    math-verify parses LaTeX more reliably when LaTeX expressions are inside $...$.
    """
    text = str(text).strip()

    if "$" in text:
        return text

    if "\\" in text:
        return f"${text}$"

    return text


def is_correct_math(prediction: str, gold: str) -> bool:
    gold_final = extract_math_final_answer(gold)
    pred_final = extract_math_final_answer(prediction)

    if HAS_MATH_VERIFY:
        try:
            parsed_gold = parse(maybe_wrap_latex(gold_final))
            parsed_pred = parse(maybe_wrap_latex(pred_final))
            return bool(verify(parsed_gold, parsed_pred))
        except Exception:
            pass

    # fallback: exact normalized string match
    return normalize_math_answer(pred_final) == normalize_math_answer(gold_final)


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

    numbers = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text.replace(",", ""))

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

    return abs(float(pred_answer) - float(gold_answer)) < 1e-6


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