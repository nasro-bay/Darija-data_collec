#!/usr/bin/env python
"""Tokenizes ../data/augmented_corpus.jsonl with the chosen subword
tokenizer, splits it into train/dev/test, and writes the kenlm-ready text
files (one doc per line, subword pieces space-separated -- lmplz treats
each line as an independent sentence, adding <s>/</s> itself, matching
plan.md section 5's "each comment/reply treated as an independent
sequence, not concatenated").

Tokenizer choice (plan.md section 4 asks to "read the statistics and
evaluation to choose the best one" -- see Tokenization/data/eval_results.json):
**SentencePiece Unigram @ 20,000 vocab**. Among the non-broken deterministic
options at this project's target vocab range (16-24K per Project_context.md's
hardware note), it has the best compression (CF=0.2744, vs BPE's 0.3385)
and zero round-trip mismatches (unlike WordPiece's 78/1,926 newline-related
mismatches, see Tokenization/PLAN.md). Byte-fallback also means it never
truly OOVs, which matters for a smoothed count model.

Word-level n-gram order, translated to subword-token order (plan.md
section 3): the plan's own worked example assumes "~3 subword tokens per
word" to justify a 9-token window for a word-trigram. That assumption
does NOT hold for the tokenizer actually chosen here -- Unigram @ 20K's
measured fertility is 1.4471 tokens/word (see eval_results.json), not 3.
This script recomputes the real subword order needed to span N whole
words as `ceil(N * fertility)`, rather than reusing the plan's literal
"9", and reports both the assumed and measured fertility so the
discrepancy is visible rather than silently baked in.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
N_GRAM_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = N_GRAM_DIR / "data" / "augmented_corpus.jsonl"
OUT_DIR = N_GRAM_DIR / "data"

sys.path.insert(0, str(ROOT / "Tokenization"))
from tokenizer_utils import load_tokenizer  # noqa: E402

TOKENIZER_KEY = "unigram"
VOCAB_SIZE = 20_000
MEASURED_FERTILITY = 1.4471  # Tokenization/data/eval_results.json, unigram @ 20,000

HELDOUT_PER_BUCKET = 2_000  # matches this project's established ~2,000-doc held-out convention
SPLIT_SEED = 42
BUCKETS = ("arabic", "latin", "mixed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenize and split the N-gram training corpus")
    parser.add_argument(
        "--heldout-per-bucket",
        type=int,
        default=HELDOUT_PER_BUCKET,
        help=f"dev/test docs per script bucket (default: {HELDOUT_PER_BUCKET}, split evenly dev/test)",
    )
    args = parser.parse_args()

    if not INPUT_PATH.exists():
        raise SystemExit(f"{INPUT_PATH} not found -- run build_augmented_corpus.py first.")

    print(f"Loading tokenizer: {TOKENIZER_KEY} @ {VOCAB_SIZE:,} ...")
    tok = load_tokenizer(TOKENIZER_KEY, VOCAB_SIZE)

    print("Reading augmented corpus...")
    docs_by_bucket: dict[str, list[dict]] = defaultdict(list)
    with INPUT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            doc = json.loads(line)
            docs_by_bucket[doc["script_bucket"]].append(doc)

    rng = random.Random(SPLIT_SEED)
    heldout_half = args.heldout_per_bucket // 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUT_DIR / "train.txt"

    report = {
        "tokenizer": {"key": TOKENIZER_KEY, "vocab_size": VOCAB_SIZE},
        "buckets": {},
    }

    total_train_docs = 0
    total_train_words = 0
    total_train_tokens = 0

    with train_path.open("w", encoding="utf-8") as train_f:
        for bucket in BUCKETS:
            docs = docs_by_bucket.get(bucket, [])
            real_docs = [d for d in docs if not d["is_transliterated"]]
            synthetic_docs = [d for d in docs if d["is_transliterated"]]

            # Held-out dev/test come ONLY from real (non-synthetic) docs --
            # plan.md section 5: "never evaluate on synthetic data".
            rng.shuffle(real_docs)
            dev_docs = real_docs[:heldout_half]
            test_docs = real_docs[heldout_half : args.heldout_per_bucket]
            heldout_ids = {d["id"] for d in dev_docs} | {d["id"] for d in test_docs}
            train_docs = [d for d in real_docs if d["id"] not in heldout_ids] + synthetic_docs

            for split_name, split_docs in (("dev", dev_docs), ("test", test_docs)):
                split_path = OUT_DIR / f"{split_name}_{bucket}.txt"
                with split_path.open("w", encoding="utf-8") as out_f:
                    for d in split_docs:
                        pieces = tok.pieces(d["text"])
                        out_f.write(" ".join(pieces) + "\n")

            bucket_train_words = 0
            bucket_train_tokens = 0
            for d in train_docs:
                pieces = tok.pieces(d["text"])
                train_f.write(" ".join(pieces) + "\n")
                bucket_train_words += len(d["text"].split())
                bucket_train_tokens += len(pieces)

            total_train_docs += len(train_docs)
            total_train_words += bucket_train_words
            total_train_tokens += bucket_train_tokens

            report["buckets"][bucket] = {
                "total_docs": len(docs),
                "real_docs": len(real_docs),
                "synthetic_docs": len(synthetic_docs),
                "train_docs": len(train_docs),
                "dev_docs": len(dev_docs),
                "test_docs": len(test_docs),
                "train_fertility": (bucket_train_tokens / bucket_train_words) if bucket_train_words else 0.0,
            }
            print(
                f"  {bucket}: {len(docs):,} total ({len(synthetic_docs):,} synthetic) -> "
                f"{len(train_docs):,} train / {len(dev_docs):,} dev / {len(test_docs):,} test"
            )

    overall_fertility = (total_train_tokens / total_train_words) if total_train_words else 0.0
    report["overall_train_fertility"] = overall_fertility
    report["assumed_fertility_per_plan"] = 3.0

    print(f"\nMeasured overall fertility on this corpus: {overall_fertility:.4f} tokens/word")
    print("(plan.md section 3 assumed ~3.0 tokens/word to derive a 9-token context window;")
    print(" this tokenizer's real fertility is much lower, so the effective subword order")
    print(" for an N-word context is recomputed below rather than reusing that assumption.)")

    report["subword_order_for_word_ngram"] = {}
    for n_words in (3, 4):
        order = math.ceil(n_words * overall_fertility)
        report["subword_order_for_word_ngram"][str(n_words)] = order
        print(f"  {n_words}-word context -> subword order {order} (ceil({n_words} * {overall_fertility:.4f}))")

    report_path = N_GRAM_DIR / "data" / "ngram_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote report -> {report_path}")
    print(f"Wrote train corpus ({total_train_docs:,} docs) -> {train_path}")


if __name__ == "__main__":
    main()
