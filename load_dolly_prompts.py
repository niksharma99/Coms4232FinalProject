#!/usr/bin/env python3
"""Create a small instruction-prompt file from Databricks Dolly 15k."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def make_prompt(instruction: str, context: str) -> str:
    instruction = normalize_text(instruction)
    context = normalize_text(context)
    if context:
        return f"{instruction}\n\nContext: {context}"
    return instruction


def is_usable_prompt(prompt: str, min_words: int, max_words: int) -> bool:
    if not prompt:
        return False
    words = prompt.split()
    if len(words) < min_words or len(words) > max_words:
        return False
    if sum(ch.isalpha() for ch in prompt) < 0.5 * len(prompt):
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Dolly 15k and save cleaned prompts as CSV."
    )
    parser.add_argument("--num-prompts", type=int, default=50)
    parser.add_argument("--min-words", type=int, default=6)
    parser.add_argument("--max-words", type=int, default=80)
    parser.add_argument("--output", default="dolly_prompts.csv")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Install it with `pip install datasets`."
        ) from exc

    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

    prompts: list[str] = []
    seen: set[str] = set()
    for row in dataset:
        prompt = make_prompt(row.get("instruction", ""), row.get("context", ""))
        if not is_usable_prompt(prompt, args.min_words, args.max_words):
            continue
        if prompt in seen:
            continue
        seen.add(prompt)
        prompts.append(prompt)
        if len(prompts) >= args.num_prompts:
            break

    if len(prompts) < args.num_prompts:
        raise SystemExit(
            f"Only found {len(prompts)} usable prompts; requested {args.num_prompts}."
        )

    output = Path(args.output)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt_id", "prompt"])
        writer.writeheader()
        for prompt_id, prompt in enumerate(prompts):
            writer.writerow({"prompt_id": prompt_id, "prompt": prompt})

    print(f"Wrote {len(prompts)} prompts to {output}")


if __name__ == "__main__":
    main()
