#!/usr/bin/env python
"""Trains the SentencePiece Unigram + byte-fallback tokenizer on
data/train_corpus.txt (built by build_training_corpus.py).

`normalization_rule_name="identity"` is important: SentencePiece
defaults to its own NFKC normalization pass, which would otherwise run
on top of (and potentially fight) the normalization clean_text.py
already applied to the corpus (NFKC, tachkil-stripping, presentation-
forms ligature expansion) -- disabling it means the trainer sees exactly
the corpus as cleaned, nothing silently re-normalized underneath.
"""
from __future__ import annotations

from pathlib import Path

import sentencepiece as spm

ROOT = Path(__file__).resolve().parents[1]  # scripts/ -> Tokenization/
TRAIN_CORPUS = ROOT / "data" / "train_corpus.txt"
MODEL_DIR = ROOT / "models" / "sentencepiece"
MODEL_PREFIX = MODEL_DIR / "darija_unigram"

VOCAB_SIZE = 20_000


def main() -> None:
    if not TRAIN_CORPUS.exists():
        raise FileNotFoundError(f"{TRAIN_CORPUS} not found — run build_training_corpus.py first.")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    spm.SentencePieceTrainer.train(
        input=str(TRAIN_CORPUS),
        model_prefix=str(MODEL_PREFIX),
        vocab_size=VOCAB_SIZE,
        model_type="unigram",
        byte_fallback=True,
        character_coverage=0.9995,
        normalization_rule_name="identity",
        input_sentence_size=5_000_000,
        shuffle_input_sentence=True,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
    )

    print(f"saved {MODEL_PREFIX}.model / {MODEL_PREFIX}.vocab")


if __name__ == "__main__":
    main()
