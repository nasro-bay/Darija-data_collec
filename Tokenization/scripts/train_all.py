#!/usr/bin/env python
"""Train all tokenizer variants across every configured vocabulary size.

Variants:
  1. SentencePiece Unigram
  2. Unigram + Subword Regularization
     — reuses the Unigram models; applied at encode time
  3. WordPiece
  4. Byte-level BPE

Vocab sizes:
  1_000, 5_000, 10_000, 20_000, 30_000

Run build_training_corpus.py first.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))


from tokenizer_utils import VOCAB_SIZES  # noqa: E402

from train_bpe import train as train_bpe  # noqa: E402
from train_sentencepiece import train as train_sp  # noqa: E402
from train_wordpiece import train as train_wordpiece  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train all Darija tokenizers"
    )

    parser.add_argument(
        "--vocab-sizes",
        type=int,
        nargs="+",
        default=VOCAB_SIZES,
        help=f"vocab sizes to train (default: {VOCAB_SIZES})",
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip a model if its output file already exists",
    )

    args = parser.parse_args()

    # ---------------------------------------------------------------
    # Check training corpus
    # ---------------------------------------------------------------

    train_corpus = ROOT / "data" / "train_corpus.txt"

    if not train_corpus.exists():
        raise FileNotFoundError(
            f"{train_corpus} not found — "
            "run build_training_corpus.py first."
        )

    # ---------------------------------------------------------------
    # Train each tokenizer for every vocabulary size
    # ---------------------------------------------------------------

    for vocab_size in args.vocab_sizes:

        # ===========================================================
        # 1. SentencePiece Unigram
        # ===========================================================

        unigram_out = (
            ROOT
            / "models"
            / "sentencepiece"
            / f"unigram_{vocab_size}.model"
        )

        if args.skip_existing and unigram_out.exists():
            print(f"skip existing {unigram_out.name}")
        else:
            print(
                f"\n=== SentencePiece Unigram "
                f"vocab={vocab_size:,} ==="
            )

            train_sp(vocab_size=vocab_size)

        # ===========================================================
        # 2. WordPiece
        # ===========================================================

        wordpiece_out = (
            ROOT
            / "models"
            / "wordpiece"
            / f"wordpiece_{vocab_size}"
            / "tokenizer.json"
        )

        if args.skip_existing and wordpiece_out.exists():
            print(f"skip existing wordpiece_{vocab_size}")
        else:
            print(
                f"\n=== WordPiece "
                f"vocab={vocab_size:,} ==="
            )

            train_wordpiece(vocab_size=vocab_size)

        # ===========================================================
        # 3. Byte-level BPE
        # ===========================================================

        bpe_out = (
            ROOT
            / "models"
            / "bpe"
            / f"bpe_{vocab_size}"
            / "tokenizer.json"
        )

        if args.skip_existing and bpe_out.exists():
            print(f"skip existing bpe_{vocab_size}")
        else:
            print(
                f"\n=== Byte-level BPE "
                f"vocab={vocab_size:,} ==="
            )

            train_bpe(vocab_size=vocab_size)

    # ---------------------------------------------------------------
    # Done
    # ---------------------------------------------------------------

    print("\nAll requested tokenizer models trained.")

    print(
        "Note: Unigram + Subword Regularization does not require "
        "separate training. It reuses the trained Unigram models "
        "at encoding time."
    )


if __name__ == "__main__":
    main()