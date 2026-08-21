#!/usr/bin/env python
"""Trains the KenLM n-gram model(s) on ../data/train.txt via `lmplz`
(Modified Kneser-Ney smoothing, kenlm's default) then converts each ARPA
file to KenLM's fast binary format via `build_binary` (see plan.md
section 5).

Requires the actual kenlm C++ command-line tools (`lmplz`, `build_binary`)
on PATH -- these are a separate build from the `kenlm` Python package
(`pip install kenlm` only builds the Python scoring bindings used by
evaluate_ngram.py, not these CLI tools). Build them from
https://github.com/kpu/kenlm (cmake + a C++ compiler) if they're missing.

Trains two orders, per plan.md section 5 ("3-gram ... with 4-gram as a
comparison run"): the subword order needed to span a 3-word and a 4-word
context respectively, as computed by prepare_ngram_data.py (see its
docstring for why this is NOT a literal order-3/order-4 model -- the
model operates over subword tokens, so reaching a 3-*word* effective
context needs a higher subword order, read from data/ngram_report.json
rather than hardcoded here).
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

N_GRAM_DIR = Path(__file__).resolve().parents[1]
TRAIN_PATH = N_GRAM_DIR / "data" / "train.txt"
REPORT_PATH = N_GRAM_DIR / "data" / "ngram_report.json"
MODELS_DIR = N_GRAM_DIR / "models"


def check_tools() -> None:
    missing = [name for name in ("lmplz", "build_binary") if shutil.which(name) is None]
    if missing:
        raise SystemExit(
            f"Missing KenLM CLI tool(s) on PATH: {', '.join(missing)}. "
            "These come from building kenlm's C++ project directly "
            "(https://github.com/kpu/kenlm) -- `pip install kenlm` alone "
            "only provides the Python scoring bindings used by "
            "evaluate_ngram.py, not lmplz/build_binary."
        )


def train_one(order: int, label: str) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    arpa_path = MODELS_DIR / f"darija_{label}.arpa"
    binary_path = MODELS_DIR / f"darija_{label}.binary"

    print(f"\n=== Training {label} (subword order={order}, Modified Kneser-Ney) ===")
    with TRAIN_PATH.open("r", encoding="utf-8") as train_f, arpa_path.open(
        "w", encoding="utf-8"
    ) as arpa_f:
        subprocess.run(
            ["lmplz", "-o", str(order), "--discount_fallback"],
            stdin=train_f,
            stdout=arpa_f,
            check=True,
        )
    print(f"  wrote {arpa_path}")

    subprocess.run(["build_binary", str(arpa_path), str(binary_path)], check=True)
    print(f"  wrote {binary_path}")


def main() -> None:
    check_tools()

    if not TRAIN_PATH.exists():
        raise SystemExit(f"{TRAIN_PATH} not found -- run prepare_ngram_data.py first.")
    if not REPORT_PATH.exists():
        raise SystemExit(f"{REPORT_PATH} not found -- run prepare_ngram_data.py first.")

    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    orders = report["subword_order_for_word_ngram"]

    train_one(orders["3"], "trigram")  # primary model, per plan.md section 5
    train_one(orders["4"], "4gram")  # comparison run


if __name__ == "__main__":
    main()
