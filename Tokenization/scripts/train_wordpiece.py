from __future__ import annotations

import argparse
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordPieceTrainer

ROOT = Path(__file__).resolve().parents[1]

TRAIN_CORPUS = ROOT / "data" / "train_corpus.txt"
MODEL_DIR = ROOT / "models" / "wordpiece"


SPECIAL_TOKENS = [
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
]


def train(*, vocab_size: int) -> None:
    """Train a WordPiece tokenizer on the Darija training corpus."""

    if not TRAIN_CORPUS.exists():
        raise FileNotFoundError(
            f"{TRAIN_CORPUS} not found — "
            "run build_training_corpus.py first."
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    output_dir = MODEL_DIR / f"wordpiece_{vocab_size}"
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(
        WordPiece(
            unk_token="[UNK]",
            continuing_subword_prefix="##",
        )
    )

    tokenizer.pre_tokenizer = Whitespace()

    trainer = WordPieceTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        show_progress=True,
    )

    print(
        f"\n=== Training WordPiece vocab={vocab_size:,} ==="
    )

    tokenizer.train(
        files=[str(TRAIN_CORPUS)],
        trainer=trainer,
    )

    tokenizer.save(
        str(output_dir / "tokenizer.json")
    )

    print(
        f"saved {output_dir / 'tokenizer.json'}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a WordPiece tokenizer"
    )

    parser.add_argument(
        "--vocab-size",
        type=int,
        default=20_000,
        help="vocabulary size (default: 20,000)",
    )

    args = parser.parse_args()

    train(vocab_size=args.vocab_size)


if __name__ == "__main__":
    main()