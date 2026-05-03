#!/usr/bin/env python3
"""Create a small prompt file from WikiText for speculative decoding experiments."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def normalize_line(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    return text


def is_usable_prompt(text: str, min_words: int, max_words: int) -> bool:
    if not text:
        return False
    if text.startswith("=") and text.endswith("="):
        return False
    words = text.split()
    if len(words) < min_words or len(words) > max_words:
        return False
    if sum(ch.isalpha() for ch in text) < 0.6 * len(text):
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download WikiText and save cleaned prompts as CSV."
    )
    parser.add_argument("--dataset", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-prompts", type=int, default=50)
    parser.add_argument("--min-words", type=int, default=12)
    parser.add_argument("--max-words", type=int, default=60)
    parser.add_argument("--output", default="wikitext_prompts.csv")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Install it with `pip install datasets`."
        ) from exc

    dataset = load_dataset("wikitext", args.dataset, split=args.split)

    prompts: list[str] = []
    seen: set[str] = set()
    for row in dataset:
        prompt = normalize_line(row.get("text", ""))
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
