#!/usr/bin/env python
"""Builds the word-cluster dialect classifier's model artifact from
data/train.jsonl (see common.py's docstring for the full algorithm).

Steps:
1. Split train rows into two script groups by `script_of(clean_for_
   classification(text))`: "arabic" rows (label in {msa, darija}) and
   "latin" rows (label in {arabize, french, english}). Rows with script
   "mixed"/"other" are skipped entirely -- their labels (code_switch/
   other) are decided deterministically by the script rule at classify
   time, never by word clusters, so they contribute no training signal
   here. This mirrors label_dataset.py's own class restriction exactly.

   Clustering is done SEPARATELY per script group (not one joint K-Means
   over all words) -- deliberate: a joint clustering would spend cluster
   capacity re-discovering the arabic/latin split the regex rule already
   gives for free, instead of spending it on the actual dialect signal
   *within* each script regime (msa vs darija; arabize vs french vs
   english).

2. Word-label tally: for each row, take its *unique* whitespace-split
   words (clean_for_classification'd text) and increment
   tally[word][row_label] += 1 once per row the word appears in -- not
   once per raw token occurrence within the row. This avoids one row
   repeating a word many times dominating the tally over many rows each
   contributing it once; a word appearing in different rows with
   different labels naturally accumulates counts in multiple classes
   (exactly the "increment in both classes" case).

3. Word vectors: every unique word in a script group's vocabulary gets
   one vector via common.get_word_vector (mean of its BPE pieces'
   embeddings from the trained CBOWAttention model, same convention as
   Embeddings/word2vec_attention/word2vec_eval.ipynb).

4. K-Means(k) per script group on those vectors. Each cluster's label
   distribution = the summed word tallies of its member words, L1-
   normalized into a probability vector over that group's classes.

5. "Verify the assumption" diagnostic: for a small sweep of k values,
   report each clustering's weighted-average cluster purity (the
   frequency-weighted mean of each cluster's max-class probability) --
   how much a cluster's dominant class actually dominates it, weighted by
   how many words are in that cluster. Purity near chance (1/num_classes)
   means clusters aren't separating dialect at all; purity near 1.0 means
   clusters are close to single-class. Printed for k in K_SWEEP; the
   actual model artifact is built at K_FINAL.

Run via the GPU venv's Python (loads the torch checkpoint):

    ".../ai-gpu/Scripts/python.exe" build_clusters.py
"""
from __future__ import annotations

import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    ARABIC_CLASSES, DATA_DIR, LATIN_CLASSES, MODELS_DIR,
    clean_for_classification, load_embedder, script_of, words_of,
)

K_SWEEP = [5, 10, 20, 40, 80]
K_FINAL_DEFAULT = 20
SEED = 42


def load_train_rows() -> list[dict]:
    import json
    path = DATA_DIR / "train.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} not found -- run split_dataset.py first.")
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_word_tallies(rows: list[dict], classes: list[str]) -> dict[str, dict[str, int]]:
    """word -> {label: count}, counted once per row the word appears in
    (see module docstring, step 2). Only rows whose label is in `classes`
    are considered (already true by construction given the script split,
    kept as an explicit guard)."""
    tally: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        if row["label"] not in classes:
            continue
        cleaned = clean_for_classification(row["text"])
        for word in set(words_of(cleaned)):
            tally[word][row["label"]] += 1
    return tally


def vectorize_vocab(tally: dict[str, dict[str, int]], get_word_vector) -> tuple[list[str], np.ndarray]:
    words, vecs = [], []
    skipped = 0
    for word in tally:
        v = get_word_vector(word)
        if v is None:
            skipped += 1
            continue
        words.append(word)
        vecs.append(v)
    if skipped:
        print(f"    (skipped {skipped} words with no BPE-encodable vector)")
    return words, np.stack(vecs)


def cluster_purity(labels: np.ndarray, kmeans_labels: np.ndarray, tally: dict[str, dict[str, int]],
                    words: list[str], classes: list[str], k: int) -> float:
    """Frequency-weighted mean of each cluster's max-class probability
    (see module docstring, step 5). `labels`/`kmeans_labels` unused
    directly here -- distributions are recomputed straight from `tally`
    grouped by kmeans_labels, same as the real model-building path, so
    this metric reflects exactly what the final artifact would contain.
    """
    cluster_dist = [defaultdict(float) for _ in range(k)]
    cluster_weight = [0.0 for _ in range(k)]
    for word, cid in zip(words, kmeans_labels):
        counts = tally[word]
        total = sum(counts.values())
        if total == 0:
            continue
        for cls in classes:
            cluster_dist[cid][cls] += counts.get(cls, 0)
        cluster_weight[cid] += total

    weighted_purity_sum = 0.0
    total_weight = sum(cluster_weight)
    for cid in range(k):
        w = cluster_weight[cid]
        if w == 0:
            continue
        max_count = max(cluster_dist[cid].values())
        purity = max_count / w
        weighted_purity_sum += purity * w
    return weighted_purity_sum / total_weight if total_weight else 0.0


def build_group_clusters(rows: list[dict], classes: list[str], group_name: str, get_word_vector, k_final: int) -> dict:
    print(f"\n=== {group_name} (classes: {classes}) ===")
    tally = build_word_tallies(rows, classes)
    print(f"  {len(tally):,} unique words tallied")
    words, X = vectorize_vocab(tally, get_word_vector)
    print(f"  {X.shape[0]:,} words vectorized ({X.shape[1]} dims)")

    print(f"  Purity sweep (chance level = {1 / len(classes):.3f}):")
    for k in K_SWEEP:
        if k >= X.shape[0]:
            continue
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(X)
        purity = cluster_purity(None, km.labels_, tally, words, classes, k)
        print(f"    k={k:<4d} weighted purity={purity:.3f}")

    print(f"  Building final model at k={k_final}...")
    km_final = KMeans(n_clusters=k_final, random_state=SEED, n_init=10).fit(X)

    cluster_dist = [defaultdict(float) for _ in range(k_final)]
    for word, cid in zip(words, km_final.labels_):
        counts = tally[word]
        for cls in classes:
            cluster_dist[cid][cls] += counts.get(cls, 0)
    normalized_dist = []
    for cid in range(k_final):
        total = sum(cluster_dist[cid].values())
        normalized_dist.append(
            {cls: (cluster_dist[cid].get(cls, 0) / total if total else 0.0) for cls in classes}
        )

    return {
        "classes": classes,
        "centroids": km_final.cluster_centers_,  # (k_final, embed_dim)
        "cluster_label_dist": normalized_dist,  # list[dict[class -> prob]], len k_final
        "k": k_final,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=K_FINAL_DEFAULT, help="clusters per script group for the final model")
    parser.add_argument("--embedder", choices=["attention", "cbow"], default="attention")
    args = parser.parse_args()

    rows = load_train_rows()
    print(f"Loaded {len(rows):,} training rows")

    get_word_vector, checkpoint_path = load_embedder(args.embedder)
    print(f"Embedder: {args.embedder} ({checkpoint_path.name})")

    arabic_rows = [r for r in rows if script_of(clean_for_classification(r["text"])) == "arabic"]
    latin_rows = [r for r in rows if script_of(clean_for_classification(r["text"])) == "latin"]
    print(f"arabic-script rows: {len(arabic_rows):,}  latin-script rows: {len(latin_rows):,}")

    arabic_model = build_group_clusters(arabic_rows, ARABIC_CLASSES, "arabic", get_word_vector, args.k)
    latin_model = build_group_clusters(latin_rows, LATIN_CLASSES, "latin", get_word_vector, args.k)

    artifact = {
        "embedder": args.embedder,
        "checkpoint": str(checkpoint_path),
        "arabic": arabic_model,
        "latin": latin_model,
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / "word_clusters.pkl"
    with out_path.open("wb") as f:
        pickle.dump(artifact, f)
    print(f"\nSaved model artifact to {out_path}")


if __name__ == "__main__":
    main()
