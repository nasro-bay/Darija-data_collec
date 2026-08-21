#!/usr/bin/env python
"""Evaluates a trained KenLM model against the held-out dev/test sets
(see plan.md section 6):

- Perplexity, computed separately per script bucket (arabic / latin /
  mixed) on both dev and test, so augmentation's effect on Arabizi-side
  perplexity can be checked without conflating it with Arabic-script
  performance.
- OOV rate at the subword-token level, from the LM's own trained
  vocabulary (distinct from the *tokenizer's* vocabulary -- byte-fallback
  SentencePiece Unigram never truly OOVs at the tokenizer level, see
  Tokenization/PLAN.md, but the n-gram model's vocabulary is whatever
  subword pieces actually appeared often enough in training, which is a
  different and meaningful "did the LM ever see this" signal).
- Pieces-per-word check: re-reports the measured fertility (already
  computed once in prepare_ngram_data.py on the training data) against
  each held-out split, to confirm the assumption stays stable out of
  sample.

Uses the `kenlm_query` CLI tool (built from kenlm's C++ source alongside
lmplz/build_binary -- see train_ngram.py's docstring) rather than the
`kenlm` Python package: the PyPI `kenlm` package's bundled Cython
bindings use internal CPython C-API symbols (`_PyGC_FINALIZED`,
`_PyDict_SetItem_KnownHash`, etc.) that were removed in Python 3.13,
so it fails to build here. `query` is a separate, pure-C++ CLI target
with no Python dependency at all, so it's unaffected. One whole split
file is piped through a single `kenlm_query` invocation (not one process
per document) for speed.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

N_GRAM_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = N_GRAM_DIR / "data"
MODELS_DIR = N_GRAM_DIR / "models"
BUCKETS = ("arabic", "latin", "mixed")

_TRAILER_RE = re.compile(r"Total:\s*(-?[\d.]+)\s+OOV:\s*(\d+)")


def word_count(tokenized_line: str) -> int:
    """SentencePiece Unigram marks word starts with '▁' -- counting them
    recovers the original whitespace-word count from the piece stream
    without needing the untokenized text alongside it.
    """
    pieces = tokenized_line.split()
    return sum(1 for p in pieces if p.startswith("▁")) or len(pieces)


def evaluate_split(model_path: Path, split_path: Path) -> dict:
    if not split_path.exists():
        return {"docs": 0}

    lines = [line.rstrip("\n") for line in split_path.open("r", encoding="utf-8") if line.strip()]
    if not lines:
        return {"docs": 0}

    # check=False: kenlm_query reliably produces all N output lines correctly,
    # then hits a benign Windows-only pipe-closure error ("WindowsException:
    # the pipe has been ended") on a trailing read attempt after Python
    # closes its end of stdin -- exit code is non-zero even though the data
    # is complete. Verified below by checking output line count instead of
    # trusting the exit code.
    result = subprocess.run(
        ["kenlm_query", str(model_path)],
        input="\n".join(lines) + "\n",
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    output_lines = result.stdout.splitlines()
    if len(output_lines) < len(lines):
        raise RuntimeError(
            f"kenlm_query only produced {len(output_lines)} output lines for "
            f"{len(lines)} input lines from {split_path} -- a real failure, "
            f"not the benign trailing pipe-closure one. stderr:\n{result.stderr}"
        )

    total_docs = 0
    total_words = 0
    total_pieces = 0
    total_log10_prob = 0.0
    total_oov_tokens = 0

    # kenlm_query emits one line per input sentence (in order), then a
    # trailing summary block ("Perplexity including OOVs:" etc.) -- only
    # the first len(lines) output lines are per-sentence data.
    for tokenized_line, output_line in zip(lines, output_lines):
        match = _TRAILER_RE.search(output_line)
        if not match:
            continue  # e.g. the "This binary file contains ..." banner line, if present
        total_docs += 1
        total_log10_prob += float(match.group(1))
        total_oov_tokens += int(match.group(2))
        pieces = tokenized_line.split()
        total_pieces += len(pieces)
        total_words += word_count(tokenized_line)

    if total_words == 0:
        return {"docs": 0}

    # Perplexity normalized per WORD (not per subword token), so numbers
    # are comparable across tokenizers/orders and to plan.md section 6's
    # framing -- kenlm's own reported "Perplexity including OOVs" instead
    # normalizes per model-unit (subword tokens + </s>), which isn't what
    # we want to report here.
    avg_log10_prob_per_word = total_log10_prob / total_words
    perplexity_per_word = 10 ** (-avg_log10_prob_per_word)

    return {
        "docs": total_docs,
        "words": total_words,
        "pieces": total_pieces,
        "fertility": total_pieces / total_words,
        "perplexity_per_word": perplexity_per_word,
        "oov_token_rate": total_oov_tokens / total_pieces,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained N-gram model")
    parser.add_argument(
        "--label",
        default="trigram",
        help="model label to evaluate (matches train_ngram.py's darija_<label>.binary, default: trigram)",
    )
    parser.add_argument(
        "--split",
        choices=("dev", "test"),
        default="test",
        help="which held-out split to evaluate (default: test)",
    )
    args = parser.parse_args()

    if shutil.which("kenlm_query") is None:
        raise SystemExit(
            "kenlm_query not found on PATH -- build it from kenlm's C++ source "
            "(https://github.com/kpu/kenlm, cmake target `query`), same as "
            "lmplz/build_binary."
        )

    model_path = MODELS_DIR / f"darija_{args.label}.binary"
    if not model_path.exists():
        raise SystemExit(f"{model_path} not found -- run train_ngram.py first.")

    results = {}
    for bucket in BUCKETS:
        split_path = DATA_DIR / f"{args.split}_{bucket}.txt"
        print(f"Scoring {bucket} ({args.split})...")
        results[bucket] = evaluate_split(model_path, split_path)

    print(f"\n=== {args.label} model, {args.split} split ===")
    print(f"{'bucket':<8} {'docs':>8} {'words':>10} {'fertility':>10} {'perplexity/word':>16} {'OOV token rate':>14}")
    for bucket, r in results.items():
        if r["docs"] == 0:
            print(f"{bucket:<8} (no held-out docs)")
            continue
        print(
            f"{bucket:<8} {r['docs']:>8,} {r['words']:>10,} {r['fertility']:>10.4f} "
            f"{r['perplexity_per_word']:>16.2f} {r['oov_token_rate']:>13.2%}"
        )

    out_path = DATA_DIR / f"eval_{args.label}_{args.split}.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
