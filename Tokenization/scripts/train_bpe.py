#!/usr/bin/env python
"""Trains the byte-level BPE tokenizer (GPT-2/RoBERTa-style) on
data/train_corpus.txt (built by build_training_corpus.py), via the HF
`tokenizers` library. Byte-level pre-tokenization means zero true OOV by
construction -- every byte maps to something in the base vocab.
"""
from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

ROOT = Path(__file__).resolve().parents[1]  # scripts/ -> Tokenization/
TRAIN_CORPUS = ROOT / "data" / "train_corpus.txt"
MODEL_DIR = ROOT / "models" / "bpe"

VOCAB_SIZE = 20_000
SPECIAL_TOKENS = ["<pad>", "<unk>", "<s>", "</s>"]


def main() -> None:
    if not TRAIN_CORPUS.exists():
        raise FileNotFoundError(f"{TRAIN_CORPUS} not found — run build_training_corpus.py first.")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet(),
    )
    tokenizer.train([str(TRAIN_CORPUS)], trainer)

    tokenizer.save(str(MODEL_DIR / "tokenizer.json"))
    tokenizer.model.save(str(MODEL_DIR))  # writes vocab.json + merges.txt

    print(f"saved {MODEL_DIR / 'tokenizer.json'}, vocab.json, merges.txt")


if __name__ == "__main__":
    main()
