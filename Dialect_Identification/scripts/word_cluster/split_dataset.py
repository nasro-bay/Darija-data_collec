#!/usr/bin/env python
"""Stratified 80/20 train/test split of data/labeled_10k.jsonl (20,000
Qwen-labeled rows) into data/train.jsonl / data/test.jsonl, stratified by
`label` so every class (including the rare `english`, 529 rows) is
represented proportionally in both splits. Deterministic (fixed seed) --
rerunning reproduces the exact same split.

Run via the base Python environment (no GPU/torch needed, pure data
shuffling):

    python split_dataset.py
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
IN_PATH = DATA_DIR / "labeled_10k.jsonl"
TRAIN_PATH = DATA_DIR / "train.jsonl"
TEST_PATH = DATA_DIR / "test.jsonl"

TEST_FRACTION = 0.2
SEED = 42


def main() -> None:
    rng = random.Random(SEED)

    with IN_PATH.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)

    train_rows: list[dict] = []
    test_rows: list[dict] = []
    for label, group in by_label.items():
        group = group[:]
        rng.shuffle(group)
        n_test = max(1, round(len(group) * TEST_FRACTION))
        test_rows.extend(group[:n_test])
        train_rows.extend(group[n_test:])
        print(f"  {label:<12} total={len(group):5d}  train={len(group) - n_test:5d}  test={n_test:5d}")

    rng.shuffle(train_rows)
    rng.shuffle(test_rows)

    with TRAIN_PATH.open("w", encoding="utf-8") as f:
        for row in train_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with TEST_PATH.open("w", encoding="utf-8") as f:
        for row in test_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(train_rows):,} rows to {TRAIN_PATH}")
    print(f"Wrote {len(test_rows):,} rows to {TEST_PATH}")


if __name__ == "__main__":
    main()
