#!/usr/bin/env python
"""Classifies text with the word-cluster model built by build_clusters.py,
and (as a script) evaluates it against data/test.jsonl.

Per-text pipeline (see common.py's docstring for the full algorithm):

1. clean_for_classification(text), then script_of() on the cleaned text
   -- same deterministic gate as label_dataset.py:
   - "mixed"  -> label = code_switch directly, no clustering involved.
   - "other"  -> label = "other" directly, no clustering involved.
   - "arabic" -> only {msa, darija} are candidate labels.
   - "latin"  -> only {arabize, french, english} are candidate labels.
2. For "arabic"/"latin" rows: split into words, map each word to its
   nearest cluster centroid (by Euclidean distance in embedding space --
   including words never seen during training, since centroids/
   distributions are precomputed and any word can be embedded and
   distance-compared), sum that cluster's label-distribution vector
   across every word in the text. Argmax over the group's candidate
   classes is the prediction. (Restricting to the group's own classes is
   automatic here, not an extra masking step -- arabic-group clusters
   only ever hold {msa, darija} mass and latin-group clusters only ever
   hold {arabize, french, english} mass, by construction in
   build_clusters.py; this still matches the "must not contain non-zero
   score for [classes outside the allowed set]" requirement exactly.)
3. If a text's word list is empty after cleaning (falls back to "other").

Run via the GPU venv's Python:

    ".../ai-gpu/Scripts/python.exe" classify.py            # full test-set eval
    ".../ai-gpu/Scripts/python.exe" classify.py --limit 200 # quick smoke test
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    ALL_CONTENT_CLASSES, DATA_DIR, MODELS_DIR,
    clean_for_classification, load_embedder, script_of, words_of,
)


class WordClusterClassifier:
    def __init__(self, artifact_path: Path = MODELS_DIR / "word_clusters.pkl"):
        if not artifact_path.exists():
            raise SystemExit(f"{artifact_path} not found -- run build_clusters.py first.")
        with artifact_path.open("rb") as f:
            self.artifact = pickle.load(f)
        self.get_word_vector, _ = load_embedder(self.artifact["embedder"])

    def _score_words(self, words: list[str], group: str) -> dict[str, float]:
        model = self.artifact[group]
        centroids = model["centroids"]  # (k, dim)
        cluster_dist = model["cluster_label_dist"]  # list[dict]
        classes = model["classes"]

        scores = {cls: 0.0 for cls in classes}
        for word in words:
            v = self.get_word_vector(word)
            if v is None:
                continue
            dists = np.linalg.norm(centroids - v, axis=1)
            cid = int(np.argmin(dists))
            for cls in classes:
                scores[cls] += cluster_dist[cid][cls]
        return scores

    def classify(self, text: str) -> tuple[str, dict[str, float] | None]:
        cleaned = clean_for_classification(text)
        script = script_of(cleaned)

        if script == "mixed":
            return "code_switch", None
        if script == "other":
            return "other", None

        words = words_of(cleaned)
        if not words:
            return "other", None

        group = "arabic" if script == "arabic" else "latin"
        scores = self._score_words(words, group)
        if all(v == 0.0 for v in scores.values()):
            # No word in the text had any training-tally mass in its
            # nearest cluster -- genuinely no signal, not a silent
            # arbitrary pick between classes.
            return "parse_error", scores
        label = max(scores, key=scores.get)
        return label, scores


def load_test_rows() -> list[dict]:
    path = DATA_DIR / "test.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} not found -- run split_dataset.py first.")
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="rows to evaluate (default: full test set)")
    args = parser.parse_args()

    clf = WordClusterClassifier()
    rows = load_test_rows()
    if args.limit:
        rows = rows[: args.limit]
    print(f"Evaluating on {len(rows):,} test rows...")

    confusion: dict[str, Counter] = defaultdict(Counter)
    correct_overall = 0
    correct_scored = 0  # excludes code_switch/other (decided deterministically, not by clustering)
    n_scored = 0

    for row in rows:
        pred, _ = clf.classify(row["text"])
        true = row["label"]
        confusion[true][pred] += 1
        if pred == true:
            correct_overall += 1
        if true in ALL_CONTENT_CLASSES:
            n_scored += 1
            if pred == true:
                correct_scored += 1

    print(f"\nOverall accuracy (all rows, incl. deterministic code_switch/other): "
          f"{correct_overall / len(rows):.4f} ({correct_overall}/{len(rows)})")
    print(f"Cluster-scored accuracy (msa/darija/arabize/french/english rows only, "
          f"the actual thing being tested): {correct_scored / n_scored:.4f} ({correct_scored}/{n_scored})")

    print("\nPer-class accuracy:")
    for true_label in sorted(confusion):
        total = sum(confusion[true_label].values())
        correct = confusion[true_label][true_label]
        print(f"  {true_label:<12} {correct:4d}/{total:<4d}  acc={correct / total:.3f}")

    print("\nConfusion matrix (rows=true, cols=predicted):")
    all_labels = sorted({lbl for c in confusion.values() for lbl in c} | set(confusion.keys()))
    header = "true\\pred".ljust(14) + "".join(f"{l[:10]:>12}" for l in all_labels)
    print(header)
    for true_label in sorted(confusion):
        row_str = true_label.ljust(14) + "".join(f"{confusion[true_label][l]:>12}" for l in all_labels)
        print(row_str)


if __name__ == "__main__":
    main()
