#!/usr/bin/env python
"""Evaluate trained tokenizers: timing, Compression Factor (CF), fertility, round-trip.

CF = total effective generated tokens / (total characters + total words)
For <unk> on a word: effective cost = len(word) + 1 (otherwise cost = #pieces).

Lower CF → better compression. Higher CF → more unknowns / splitting.

Writes JSON results to data/eval_results.json by default.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tokenizer_utils import (  # noqa: E402
    TOKENIZER_LABELS,
    VOCAB_SIZES,
    classify_script,
    compression_factor,
    discover_available_models,
    fertility_for_words,
    load_heldout_docs,
    load_tokenizer,
    time_encode_decode,
)


def evaluate_one(key: str, vocab_size: int, docs: list[dict], *, timing_rounds: int) -> dict:
    tok = load_tokenizer(key, vocab_size)
    texts = [d["text"] for d in docs]
    words = [(w, classify_script(w)) for t in texts for w in t.split() if w.strip()]

    timing = time_encode_decode(tok, texts, rounds=timing_rounds)
    cf_values = [compression_factor(t, tok.pieces) for t in texts]
    fert = fertility_for_words(words, tok.pieces)

    mismatches = []
    for t in texts:
        decoded = tok.decode(tok.encode(t))
        if decoded != t:
            mismatches.append(t[:120])

    return {
        "tokenizer_key": key,
        "tokenizer_label": TOKENIZER_LABELS.get(key, key),
        "vocab_size": vocab_size,
        "vocab_size_actual": tok.vocab_size_actual,
        "heldout_docs": len(docs),
        "timing": timing,
        "compression_factor_mean": sum(cf_values) / len(cf_values) if cf_values else 0.0,
        "compression_factor_median": sorted(cf_values)[len(cf_values) // 2] if cf_values else 0.0,
        "fertility": fert,
        "roundtrip_mismatches": len(mismatches),
        "roundtrip_mismatch_samples": mismatches[:5],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Darija tokenizers")
    parser.add_argument(
        "--vocab-sizes",
        type=int,
        nargs="+",
        default=None,
        help="vocab sizes to evaluate (default: all trained on disk, else VOCAB_SIZES)",
    )
    parser.add_argument(
        "--timing-rounds",
        type=int,
        default=3,
        help="passes over held-out set for timing averages",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "eval_results.json",
        help="JSON output path",
    )
    args = parser.parse_args()

    docs = load_heldout_docs()
    available = discover_available_models()

    if args.vocab_sizes:
        requested = {(k, v) for k in ("unigram", "unigram_sr", "wordpiece", "bpe") for v in args.vocab_sizes}
        pairs = sorted(requested & set(available), key=lambda x: (x[1], x[0]))
        missing = requested - set(available)
        for key, vs in sorted(missing):
            print(f"warning: no trained model for {key} vocab={vs:,}", file=sys.stderr)
    else:
        pairs = available

    if not pairs:
        raise FileNotFoundError("no trained models found — run scripts/train_all.py first")

    results = []
    for key, vocab_size in pairs:
        print(f"evaluating {TOKENIZER_LABELS.get(key, key)} @ {vocab_size:,} …")
        try:
            results.append(evaluate_one(key, vocab_size, docs, timing_rounds=args.timing_rounds))
        except FileNotFoundError as exc:
            print(f"  skip: {exc}", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "heldout_docs": len(docs),
        "vocab_sizes_configured": VOCAB_SIZES,
        "results": results,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {len(results)} result rows -> {args.output}")

    # Summary table
    print(f"\n{'tokenizer':<35} {'vocab':>7} {'CF':>8} {'fertility':>10} {'enc ms/doc':>12}")
    print("-" * 80)
    for row in results:
        print(
            f"{row['tokenizer_label']:<35} {row['vocab_size']:>7,} "
            f"{row['compression_factor_mean']:>8.4f} "
            f"{row['fertility']['overall_fertility']:>10.4f} "
            f"{row['timing']['encode_ms_per_doc']:>12.4f}"
        )


if __name__ == "__main__":
    main()
