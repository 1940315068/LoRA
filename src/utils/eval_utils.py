import re
from typing import Dict, Optional
import torch
import os
import sys
import tempfile
import subprocess


NUMBER_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

try:
    from math_verify import parse, verify
    HAS_MATH_VERIFY = True
except Exception:
    HAS_MATH_VERIFY = False


def extract_python_code(text: str) -> Optional[str]:
    """
    Extract Python code from model output.

    Priority:
    1. ```python ... ```
    2. ``` ... ```
    3. whole output if it looks like Python/Gurobi code
    """
    if text is None:
        return None

    text = str(text).strip()

    # Prefer ```python ... ```
    m = re.search(
        r"```python\s*(.*?)```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()

    # Fallback: any fenced code block
    m = re.search(
        r"```\s*(.*?)```",
        text,
        flags=re.DOTALL,
    )
    if m:
        return m.group(1).strip()

    # Fallback: whole output if it looks like code
    code_signals = [
        "import gurobipy",
        "from gurobipy import",
        "gp.Model",
        "Model(",
        ".optimize()",
    ]

    if any(signal in text for signal in code_signals):
        return text

    return None


def run_python_code(code: str, timeout: int = 100) -> Dict:
    """
    Run generated Python code in a temporary directory.

    Returns:
        {
            executed: bool,
            timeout: bool,
            returncode: int or None,
            stdout: str,
            stderr: str,
        }
    """
    if code is None or not str(code).strip():
        return {
            "executed": False,
            "timeout": False,
            "returncode": None,
            "stdout": "",
            "stderr": "No code to execute.",
        }

    with tempfile.TemporaryDirectory() as tmpdir:
        code_path = os.path.join(tmpdir, "solution.py")

        with open(code_path, "w", encoding="utf-8") as f:
            f.write(code)

        try:
            proc = subprocess.run(
                [sys.executable, code_path],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            return {
                "executed": proc.returncode == 0,
                "timeout": False,
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }

        except subprocess.TimeoutExpired as e:
            return {
                "executed": False,
                "timeout": True,
                "returncode": None,
                "stdout": e.stdout or "",
                "stderr": e.stderr or "Execution timed out.",
            }

        except Exception as e:
            return {
                "executed": False,
                "timeout": False,
                "returncode": None,
                "stdout": "",
                "stderr": repr(e),
            }


def extract_optimal_value(stdout: str) -> Optional[float]:
    """
    Extract optimal objective value from stdout.

    Preferred format:
        OPTIMAL_VALUE: 123.45

    Fallback:
        use the last number in stdout.
    """
    if stdout is None:
        return None

    stdout = str(stdout)

    m = re.search(
        r"OPTIMAL_VALUE\s*:\s*(" + NUMBER_RE + r")",
        stdout,
        flags=re.IGNORECASE,
    )
    if m:
        return float(m.group(1))

    nums = re.findall(NUMBER_RE, stdout)

    if not nums:
        return None

    try:
        return float(nums[-1])
    except Exception:
        return None


def is_correct_optmath(
    pred_value,
    gold_value,
    tolerance: float = 0.05,
) -> bool:
    """
    Compare predicted objective value with gold answer.

    Default tolerance is relative 5%, following OptMATH-style numerical tolerance.
    """
    if pred_value is None or gold_value is None:
        return False

    try:
        pred = float(pred_value)
        gold = float(gold_value)
    except Exception:
        return False

    return abs(pred - gold) <= tolerance * max(1.0, abs(gold))


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