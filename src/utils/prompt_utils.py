def build_qwen3_user_content(question: str) -> str:
    return (
        "Solve the following math problem step by step. "
        "Put the final numerical answer after ####.\n\n"
        f"{question}"
    )


def build_qwen3_sft_text(tokenizer, question: str, answer: str) -> str:
    messages = [
        {
            "role": "user",
            "content": build_qwen3_user_content(question),
        },
        {
            "role": "assistant",
            "content": answer,
        },
    ]

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )


def build_qwen3_eval_prompt(tokenizer, question: str) -> str:
    messages = [
        {
            "role": "user",
            "content": build_qwen3_user_content(question),
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