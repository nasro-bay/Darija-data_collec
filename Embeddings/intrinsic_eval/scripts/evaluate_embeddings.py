#!/usr/bin/env python
"""Evaluates a word-embedding model against ../data/word_similarity.jsonl
(Spearman correlation) and ../data/analogy_pairs.jsonl (vector-arithmetic
nearest-neighbor accuracy). See ../plan.md for dataset design/methodology.

Usage as a library (once a real embedding model exists):

    from evaluate_embeddings import evaluate_similarity, evaluate_analogies

    def embed(word: str):
        return my_model[word] if word in my_model else None  # -> np.ndarray | None

    sim_result = evaluate_similarity(embed)
    analogy_result = evaluate_analogies(embed, vocab=list(my_model.key_to_index))

CLI usage (no real model yet -- smoke-tests the harness itself):

    python evaluate_embeddings.py --dummy-random
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.stats import spearmanr

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

EmbedFn = Callable[[str], "np.ndarray | None"]


def load_similarity_pairs() -> list[dict]:
    path = DATA_DIR / "word_similarity.jsonl"
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_analogy_categories() -> list[dict]:
    path = DATA_DIR / "analogy_pairs.jsonl"
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def evaluate_similarity(embed: EmbedFn) -> dict:
    """Spearman correlation between human scores and model cosine
    similarities, overall and per category. Pairs where either word is
    OOV for the model are skipped and counted separately -- OOV rate is
    itself a meaningful diagnostic, not just noise to ignore silently.
    """
    pairs = load_similarity_pairs()
    human_scores: list[float] = []
    model_scores: list[float] = []
    per_category: dict[str, dict] = {}
    oov_pairs = 0

    for row in pairs:
        cat = row["category"]
        per_category.setdefault(cat, {"human": [], "model": [], "oov": 0})

        v1 = embed(row["word1"])
        v2 = embed(row["word2"])
        if v1 is None or v2 is None:
            oov_pairs += 1
            per_category[cat]["oov"] += 1
            continue

        sim = _cosine(v1, v2)
        human_scores.append(row["score"])
        model_scores.append(sim)
        per_category[cat]["human"].append(row["score"])
        per_category[cat]["model"].append(sim)

    overall_corr, overall_p = (
        spearmanr(human_scores, model_scores) if len(human_scores) >= 2 else (float("nan"), float("nan"))
    )

    category_results = {}
    for cat, d in per_category.items():
        if len(d["human"]) >= 2:
            corr, p = spearmanr(d["human"], d["model"])
        else:
            corr, p = float("nan"), float("nan")
        category_results[cat] = {
            "spearman": corr,
            "p_value": p,
            "n_pairs": len(d["human"]),
            "n_oov": d["oov"],
        }

    return {
        "overall_spearman": overall_corr,
        "overall_p_value": overall_p,
        "n_pairs_scored": len(human_scores),
        "n_pairs_oov": oov_pairs,
        "by_category": category_results,
    }


def evaluate_analogies(embed: EmbedFn, vocab: list[str] | None = None, top_k: int = 1) -> dict:
    """For every category, combines its base pairs pairwise into analogy
    questions: given (a1, b1) and (a2, b2) in the same relation category,
    predict b2 from `embed(b1) - embed(a1) + embed(a2)` and check whether
    the nearest neighbor (excluding a1, b1, a2 themselves) is b2, within
    top_k. `vocab` is the nearest-neighbor candidate pool; defaults to
    every word appearing anywhere in the analogy file if not given (fine
    for a smoke test, too small to be a meaningful accuracy number for a
    real model -- pass the model's full vocabulary for real evaluation).
    """
    categories = load_analogy_categories()

    if vocab is None:
        vocab = sorted({p["word_a"] for cat in categories for p in cat["pairs"]}
                       | {p["word_b"] for cat in categories for p in cat["pairs"]})

    vocab_vecs: dict[str, np.ndarray] = {}
    for w in vocab:
        v = embed(w)
        if v is not None:
            vocab_vecs[w] = v

    results = {}
    for cat in categories:
        name = cat["category"]
        pairs = cat["pairs"]
        correct = 0
        total = 0
        skipped_oov = 0

        for i, p1 in enumerate(pairs):
            for j, p2 in enumerate(pairs):
                if i == j:
                    continue
                a1, b1 = p1["word_a"], p1["word_b"]
                a2, b2 = p2["word_a"], p2["word_b"]

                va1, vb1, va2 = embed(a1), embed(b1), embed(a2)
                if va1 is None or vb1 is None or va2 is None:
                    skipped_oov += 1
                    continue

                target = vb1 - va1 + va2
                exclude = {a1, b1, a2}
                candidates = [(w, v) for w, v in vocab_vecs.items() if w not in exclude]
                if not candidates:
                    skipped_oov += 1
                    continue

                scored = sorted(candidates, key=lambda wv: -_cosine(target, wv[1]))
                predicted = [w for w, _ in scored[:top_k]]

                total += 1
                if b2 in predicted:
                    correct += 1

        results[name] = {
            "accuracy": (correct / total) if total else float("nan"),
            "n_questions": total,
            "n_skipped_oov": skipped_oov,
        }

    overall_correct = sum(r["accuracy"] * r["n_questions"] for r in results.values() if r["n_questions"])
    overall_total = sum(r["n_questions"] for r in results.values())

    return {
        "overall_accuracy": (overall_correct / overall_total) if overall_total else float("nan"),
        "overall_n_questions": overall_total,
        "by_category": results,
    }


def _dummy_random_embed(dim: int = 50, seed: int = 42) -> EmbedFn:
    """Deterministic random-vector 'embedding' -- used only to smoke-test
    that the harness itself runs correctly end-to-end, ahead of a real
    trained model existing. Expected result: near-zero Spearman
    correlation and near-chance analogy accuracy (random vectors carry no
    real structure) -- if this comes back suspiciously high or the script
    errors, the bug is in the harness, not a model.
    """
    rng = np.random.default_rng(seed)
    cache: dict[str, np.ndarray] = {}

    def embed(word: str) -> np.ndarray:
        if word not in cache:
            cache[word] = rng.normal(size=dim)
        return cache[word]

    return embed


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a word embedding model against the Darija intrinsic eval set")
    parser.add_argument(
        "--dummy-random",
        action="store_true",
        help="smoke-test the harness with random vectors instead of a real model",
    )
    args = parser.parse_args()

    if not args.dummy_random:
        raise SystemExit(
            "No real embedding model wired in yet -- this CLI only supports --dummy-random "
            "for now. Import evaluate_similarity()/evaluate_analogies() directly once a "
            "trained model exists."
        )

    embed = _dummy_random_embed()

    print("=== Word similarity (Spearman correlation) ===")
    sim = evaluate_similarity(embed)
    print(f"Overall: r={sim['overall_spearman']:.4f} (p={sim['overall_p_value']:.4f}), "
          f"n={sim['n_pairs_scored']}, oov={sim['n_pairs_oov']}")
    for cat, r in sim["by_category"].items():
        print(f"  {cat:<24} r={r['spearman']:.4f}  n={r['n_pairs']}  oov={r['n_oov']}")

    print("\n=== Analogies (vector-arithmetic nearest-neighbor accuracy) ===")
    ana = evaluate_analogies(embed)
    print(f"Overall: acc={ana['overall_accuracy']:.4f}, n={ana['overall_n_questions']}")
    for cat, r in ana["by_category"].items():
        print(f"  {cat:<24} acc={r['accuracy']:.4f}  n={r['n_questions']}  skipped_oov={r['n_skipped_oov']}")


if __name__ == "__main__":
    main()
