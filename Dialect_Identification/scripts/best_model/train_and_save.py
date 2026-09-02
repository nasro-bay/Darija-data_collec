#!/usr/bin/env python
"""Trains and saves the project's best dialect-ID models -- char n-gram
TF-IDF + SVM-RBF, per script group -- for real deployment use (estimating
the dialect distribution over the full YouTube corpus), not just
evaluation. See notebooks/04_baldwin_lui_ngrams.ipynb for where this
architecture/hyperparameter choice came from (best of 48+16 combos
compared there): char_trigram+SVM-RBF for the arabic group (0.833 test
accuracy), char_bigram+SVM-RBF for the latin group (0.881).

Unlike that notebook (which holds out data/test.jsonl for evaluation),
this script trains on **all 20,000 labeled rows** (train.jsonl +
test.jsonl combined) -- there's nothing left to hold out for once the
model is being deployed rather than compared against other methods, and
more real training data only helps here.

Run via the base Python environment (sklearn only, no torch needed):

    python train_and_save.py
"""
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "Dialect_Identification" / "data"
MODELS_DIR = ROOT / "Dialect_Identification" / "models" / "best_model"

SEED = 42

_MENTION_WITH_FRAGMENT_RE = re.compile(r"\[MENTION\](\s*-[^\s]{1,10})?")
_URL_RE = re.compile(r"\[URL\]")
_INLINE_WHITESPACE_RE = re.compile(r"[ \t]+")
_ARABIC_RE = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")
_LATIN_RE = re.compile(r"[a-zA-ZÀ-ɏ]")

ARABIC_CLASSES = ["msa", "darija"]
LATIN_CLASSES = ["arabize", "french", "english"]
GROUP_CLASSES = {"arabic": ARABIC_CLASSES, "latin": LATIN_CLASSES}

# Winning (group -> (n-gram order, vocab cap)) per notebooks/04_baldwin_lui_ngrams.ipynb's
# results_df: char_trigram for arabic (0.833), char_bigram for latin (0.881).
GROUP_NGRAM_ORDER = {"arabic": 3, "latin": 2}
VOCAB_CAP = 3000


def clean_for_classification(text: str) -> str:
    text = _MENTION_WITH_FRAGMENT_RE.sub("", text)
    text = _URL_RE.sub("", text)
    return _INLINE_WHITESPACE_RE.sub(" ", text).strip()


def script_of(text: str) -> str:
    has_ar = bool(_ARABIC_RE.search(text))
    has_lat = bool(_LATIN_RE.search(text))
    if has_ar and has_lat:
        return "mixed"
    if has_ar:
        return "arabic"
    if has_lat:
        return "latin"
    return "other"


def load_group_rows(group: str) -> list[dict]:
    """train.jsonl + test.jsonl combined -- see module docstring for why."""
    rows = []
    for split in ("train", "test"):
        with (DATA_DIR / f"{split}.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                cleaned = clean_for_classification(r["text"])
                if script_of(cleaned) != group:
                    continue
                rows.append({"id": r["id"], "text": r["text"], "label": r["label"]})
    return rows


def smote_balance(X: sp.csr_matrix, y: np.ndarray, k: int = 5, random_state: int = SEED):
    """Same from-scratch SMOTE as notebook 04 (imbalanced-learn isn't
    installable in this sandbox -- no outbound network)."""
    rng = np.random.RandomState(random_state)
    classes, counts = np.unique(y, return_counts=True)
    target = counts.max()

    X_parts, y_parts = [X], [y]
    for c, cnt in zip(classes, counts):
        n_needed = target - cnt
        if n_needed <= 0:
            continue
        X_c = X[y == c]
        if X_c.shape[0] < 2:
            idx = rng.randint(0, X_c.shape[0], size=n_needed)
            X_parts.append(X_c[idx])
            y_parts.append(np.full(n_needed, c))
            continue

        n_neighbors = min(k + 1, X_c.shape[0])
        nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(X_c)
        base_idx = rng.randint(0, X_c.shape[0], size=n_needed)
        _, neighbor_idx = nn_model.kneighbors(X_c[base_idx])

        synth_rows = []
        for i, b in enumerate(base_idx):
            candidates = neighbor_idx[i][1:]
            nb = candidates[rng.randint(0, len(candidates))] if len(candidates) else b
            gap = rng.uniform(0.0, 1.0)
            synth_rows.append(X_c[b] + gap * (X_c[nb] - X_c[b]))
        X_parts.append(sp.vstack(synth_rows).tocsr())
        y_parts.append(np.full(n_needed, c))

    return sp.vstack(X_parts).tocsr(), np.concatenate(y_parts)


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for group in ("arabic", "latin"):
        classes = GROUP_CLASSES[group]
        label_to_id = {c: i for i, c in enumerate(classes)}
        n = GROUP_NGRAM_ORDER[group]

        rows = load_group_rows(group)
        texts = [r["text"] for r in rows]
        y = np.array([label_to_id[r["label"]] for r in rows])
        print(f"{group}: {len(rows):,} labeled rows (train.jsonl+test.jsonl), classes={classes}")

        vec = TfidfVectorizer(analyzer="char", ngram_range=(n, n), max_features=VOCAB_CAP, min_df=2)
        X = vec.fit_transform(texts)
        X_bal, y_bal = smote_balance(X, y)
        print(f"  vocab={len(vec.vocabulary_)}  balanced={X_bal.shape[0]:,}")

        model = SVC(kernel="rbf", random_state=SEED)
        model.fit(X_bal, y_bal)
        print(f"  fit done, {model.n_support_.sum():,} support vectors")

        with (MODELS_DIR / f"{group}_vectorizer.pkl").open("wb") as f:
            pickle.dump(vec, f)
        with (MODELS_DIR / f"{group}_svm.pkl").open("wb") as f:
            pickle.dump(model, f)
        print(f"  saved to {MODELS_DIR / f'{group}_vectorizer.pkl'} / {group}_svm.pkl\n")

    print("Done.")


if __name__ == "__main__":
    main()
