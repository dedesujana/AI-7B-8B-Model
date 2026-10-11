#!/usr/bin/env python3
"""
Download and convert Indo4B dataset to plain text for training.
Indo4B is a large Indonesian text corpus.
"""
import argparse
import os
import re
from typing import Iterator
from datasets import load_dataset


def normalize_text(text: str) -> str:
    """Clean and normalize text"""
    if not isinstance(text, str):
        return ""
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def iter_indo4b_rows(dataset, min_chars: int = 20, max_chars: int = 0, max_rows: int = 0):
    """Iterate through Indo4B dataset rows"""
    count = 0
    for row in dataset:
        # Indo4B has 'text' column
        text = row.get("text") or ""
        if not isinstance(text, str):
            continue

        text = normalize_text(text)
        if len(text) < min_chars:
            continue

        if max_chars and len(text) > max_chars:
            text = text[:max_chars]

        yield text
        count += 1

        if max_rows and count >= max_rows:
            break


def build_txt_from_indo4b(
    output_path: str,
    max_rows: int = 0,
    target_mb: float = 0.0,
    min_chars: int = 20,
    max_chars: int = 0,
    split: str = "train",
):
    """Download Indo4B and save to text file"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    print(f"Loading Indo4B dataset ({split} split)...")
    try:
        dataset = load_dataset("indolem/indo4b", split=split, streaming=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Make sure you have internet connection and 'datasets' library installed")
        return

    total_bytes = 0
    written = 0

    print(f"Saving to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for text in iter_indo4b_rows(dataset, min_chars=min_chars, max_chars=max_chars, max_rows=max_rows):
            if not text:
                continue

            f.write(text)
            f.write("\n\n")
            total_bytes += len(text.encode("utf-8")) + 2
            written += 1

            # Progress indicator
            if written % 1000 == 0:
                print(f"  Written {written} texts ({total_bytes / (1024 * 1024):.2f} MB)...")

            if target_mb > 0:
                if total_bytes >= target_mb * 1024 * 1024:
                    print(f"Target reached: {total_bytes / (1024 * 1024):.2f} MB")
                    break

    print(f"\n✓ Done!")
    print(f"  Total texts: {written}")
    print(f"  File size: {total_bytes / (1024 * 1024):.2f} MB")
    print(f"  Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert Indo4B Indonesian dataset to plain text training file.")
    parser.add_argument("--output", type=str, default="train_data_indo4b.txt", help="Output .txt file path")
    parser.add_argument("--max_rows", type=int, default=0, help="Maximum rows to write. 0 = unlimited")
    parser.add_argument("--target_mb", type=float, default=500.0, help="Stop after reaching this many MB. Default 500MB")
    parser.add_argument("--min_chars", type=int, default=20, help="Minimum chars per row")
    parser.add_argument("--max_chars", type=int, default=0, help="Maximum chars per row. 0 = unlimited")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use")
    args = parser.parse_args()

    build_txt_from_indo4b(
        output_path=args.output,
        max_rows=args.max_rows,
        target_mb=args.target_mb,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        split=args.split,
    )


if __name__ == "__main__":
    main()
