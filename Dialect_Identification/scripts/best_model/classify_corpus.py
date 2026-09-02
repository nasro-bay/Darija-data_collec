#!/usr/bin/env python
"""Runs the saved best dialect-ID models (train_and_save.py) over a
5,000,000-row sample of the real YouTube corpus to estimate the
dialect/language distribution across the actual dataset -- not just the
20k labeled/test rows. Output feeds the DarijaDZ/Kaggle_DarijaDz README
tables (see update_readme_tables.py), replacing the old script-only
(Arabic/Latin/Mixed/Digits) distribution with a real dialect breakdown
(msa/darija/arabize/french/english/code_switch/other).

Sample: the first 5,000,000 lines of Data/youtube_corpus_shuffled.jsonl
-- already a random shuffle of the full corpus (see
Youtube_scrap/scripts/build_unified_dataset.py), so a prefix slice is a
valid random sample without needing to reservoir-sample again.

Parallelized across CPU cores (multiprocessing.Pool) -- a single-process
run benchmarked at ~390-900 rows/sec depending on script group (SVM-RBF
prediction cost dominates), which would take hours serially; sharding
across cores brings this down to tens of minutes.

Run via the base Python environment (sklearn only, no torch needed):

    python classify_corpus.py
"""
from __future__ import annotations

import json
import pickle
import re
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CORPUS_PATH = ROOT / "Data" / "youtube_corpus_shuffled.jsonl"
MODELS_DIR = ROOT / "Dialect_Identification" / "models" / "best_model"
OUT_PATH = ROOT / "Dialect_Identification" / "data" / "corpus_dialect_distribution.json"

SAMPLE_SIZE = 5_000_000
CHUNK_SIZE = 20_000
NUM_WORKERS = 14  # of 16 cores -- leaves headroom for the OS/other processes

_MENTION_WITH_FRAGMENT_RE = re.compile(r"\[MENTION\](\s*-[^\s]{1,10})?")
_URL_RE = re.compile(r"\[URL\]")
_INLINE_WHITESPACE_RE = re.compile(r"[ \t]+")
_ARABIC_RE = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")
_LATIN_RE = re.compile(r"[a-zA-ZÀ-ɏ]")

ARABIC_CLASSES = ["msa", "darija"]
LATIN_CLASSES = ["arabize", "french", "english"]


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


# Populated once per worker process by _init_worker -- avoids re-unpickling
# the vectorizers/models (and their thousands of support vectors) for every
# chunk, only once per process.
_MODELS: dict = {}


def _init_worker() -> None:
    for group in ("arabic", "latin"):
        with (MODELS_DIR / f"{group}_vectorizer.pkl").open("rb") as f:
            vec = pickle.load(f)
        with (MODELS_DIR / f"{group}_svm.pkl").open("rb") as f:
            model = pickle.load(f)
        _MODELS[group] = (vec, model)


def _classify_chunk(lines: list[str]) -> Counter:
    counts: Counter[str] = Counter()
    cleaned_by_group: dict[str, list[str]] = {"arabic": [], "latin": []}

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = row.get("text", "")
        if not text:
            counts["other"] += 1
            continue
        cleaned = clean_for_classification(text)
        group = script_of(cleaned)
        if group == "mixed":
            counts["code_switch"] += 1
        elif group == "other":
            counts["other"] += 1
        else:
            cleaned_by_group[group].append(text)

    for group, classes in (("arabic", ARABIC_CLASSES), ("latin", LATIN_CLASSES)):
        texts = cleaned_by_group[group]
        if not texts:
            continue
        vec, model = _MODELS[group]
        X = vec.transform(texts)
        preds = model.predict(X)
        for p in preds:
            counts[classes[p]] += 1

    return counts


def main() -> None:
    t_start = time.time()
    total_counts: Counter[str] = Counter()
    n_processed = 0

    with CORPUS_PATH.open("r", encoding="utf-8") as f, Pool(processes=NUM_WORKERS, initializer=_init_worker) as pool:

        def chunk_iter():
            chunk = []
            for i, line in enumerate(f):
                if i >= SAMPLE_SIZE:
                    return
                chunk.append(line)
                if len(chunk) >= CHUNK_SIZE:
                    yield chunk
                    chunk = []
            if chunk:
                yield chunk

        for chunk_counts in pool.imap_unordered(_classify_chunk, chunk_iter(), chunksize=1):
            total_counts.update(chunk_counts)
            n_processed += sum(chunk_counts.values())
            elapsed = time.time() - t_start
            rate = n_processed / elapsed if elapsed > 0 else 0
            eta_min = (SAMPLE_SIZE - n_processed) / rate / 60 if rate > 0 else float("nan")
            print(f"  ...{n_processed:,}/{SAMPLE_SIZE:,} ({rate:.0f} rows/s, ETA {eta_min:.1f} min)", flush=True)

    total = sum(total_counts.values())
    print(f"\nDone in {(time.time() - t_start) / 60:.1f} min. Total classified: {total:,}")
    for label, count in total_counts.most_common():
        print(f"  {label:<14} {count:>10,}  ({count / total * 100:.2f}%)")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump({
            "sample_size": total,
            "counts": dict(total_counts),
            "percentages": {k: v / total * 100 for k, v in total_counts.items()},
        }, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
