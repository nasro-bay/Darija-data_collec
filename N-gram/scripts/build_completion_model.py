#!/usr/bin/env python
"""Builds a lightweight Python-native n-gram completion model directly
from ../data/train.txt, for the interactive text-completion demo
(app.py).

This is a SEPARATE, simpler model from the properly Modified-Kneser-Ney
-smoothed KenLM model already trained by train_ngram.py and evaluated by
evaluate_ngram.py -- that one remains the "official" model for perplexity
numbers. This one exists because actual step-by-step generation (sample
a next token, extend, repeat) needs per-step access to "given this
context, what's the distribution over next tokens," which normally comes
from kenlm's Python bindings' state-transition API -- broken here by the
Python 3.13 C-API incompatibility (see evaluate_ngram.py's docstring),
and kenlm's own CLI tools (lmplz/build_binary/query) only ever score
already-complete text, never generate. Rebuilding the .arpa file to parse
by hand would mean a ~1 hour retrain; building fresh from train.txt
(which we still have) is the practical path that actually works here.

Algorithm: simple stupid-backoff trigram (subword-token level, not
word-level) -- for each line, tokens are wrapped with <s>/</s> (matching
KenLM's own sentence-boundary convention) and counted at the unigram,
bigram, and trigram level. Generation samples from the trigram
distribution for the current 2-token context if it exists, falling back
to bigram then unigram otherwise. Not smoothed the way the KenLM model
is -- good enough for an interactive demo, not a replacement for the
evaluated model.
"""
from __future__ import annotations

import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

N_GRAM_DIR = Path(__file__).resolve().parents[1]
TRAIN_PATH = N_GRAM_DIR / "data" / "train.txt"
OUTPUT_PATH = N_GRAM_DIR / "models" / "completion_model.pkl"

BOS = "<s>"
EOS = "</s>"


def main() -> None:
    if not TRAIN_PATH.exists():
        raise SystemExit(f"{TRAIN_PATH} not found -- run prepare_ngram_data.py first.")

    unigram_counts: Counter[str] = Counter()
    bigram_counts: dict[str, Counter[str]] = defaultdict(Counter)
    trigram_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)

    print(f"Reading {TRAIN_PATH} ...")
    with TRAIN_PATH.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            pieces = line.split()
            if not pieces:
                continue
            tokens = [BOS, BOS] + pieces + [EOS]

            for j in range(len(tokens)):
                unigram_counts[tokens[j]] += 1
                if j >= 1:
                    bigram_counts[tokens[j - 1]][tokens[j]] += 1
                if j >= 2:
                    trigram_counts[(tokens[j - 2], tokens[j - 1])][tokens[j]] += 1

            if i % 500_000 == 0:
                print(f"  ...{i:,} lines counted")

    print(f"Done counting: {len(unigram_counts):,} unigrams, {len(bigram_counts):,} bigram "
          f"contexts, {len(trigram_counts):,} trigram contexts")

    model = {
        "unigram_counts": unigram_counts,
        "bigram_counts": dict(bigram_counts),
        "trigram_counts": dict(trigram_counts),
        "bos": BOS,
        "eos": EOS,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {OUTPUT_PATH} ...")
    with OUTPUT_PATH.open("wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"Done! {OUTPUT_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
