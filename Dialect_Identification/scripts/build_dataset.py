#!/usr/bin/env python
"""Builds the one-time 10k-sentence pool to be annotated (by the local
Qwen3.5-4B classifier, see ../guide.md) for the 6-class dialect/language
ID task: french, arabize, msa, darija_arabic_script, english, code_switch.

Sampling design (agreed in-session, not re-derived here):
- djelfa forum posts: 1,500 examples, plain reservoir sample -- djelfa is
  mostly MSA already, so no script-based stratification needed there.
- YouTube comments: 8,500 examples, split into three equal script-based
  buckets via regex (not random sampling) so all three ~arabic/~latin/
  ~mixed regimes are represented regardless of the corpus's natural
  script mix:
    - "arabic": Arabic-script only (candidates for msa/darija_arabic_script)
    - "latin": Latin-script only (candidates for french/english/arabize)
    - "mixed": both scripts present in the same text (candidates for
      code_switch, though the model still makes the final call)
- Each bucket is oversampled (~30%) via single-pass reservoir sampling,
  then MinHash/LSH-deduped (same technique as dedup.py in
  Youtube_scrap/Mountada_djelfa_scrap -- built fresh here, not touching
  either project's persisted pipeline LSH state), then trimmed to the
  exact target count.

Bucketing uses clean_for_classification(text) (see notebooks/01_cleaning_
rules.ipynb), not raw text: the `[MENTION]`/`[URL]` anonymization
placeholders are themselves Latin-script tokens, so raw-text bucketing
mislabeled otherwise-pure-Arabic rows as "mixed" -- confirmed empirically
in that notebook, 71.6% of the original "mixed" bucket was this artifact,
not genuine code-switching. Only the "mixed" bucket was actually affected
("arabic"/"latin" can't gain a false placeholder-only script by
construction -- see the notebook's reasoning) but cleaning is applied
uniformly here for simplicity/consistency. The STORED text is still the
original, uncleaned row -- cleaning only decides the bucket, matching
this project's convention of keeping raw text and cleaning transiently at
point of use (same reasoning as the main pipelines keeping raw text for
re-tokenization).

Run via the base Python environment (no GPU needed -- this is pure data
sampling, not model inference):

    python build_dataset.py
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

from datasketch import MinHash, MinHashLSH

ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "unlabeled_10k.jsonl"

DJELFA_FILES = sorted((ROOT / "Mountada_djelfa_scrap" / "data" / "processed").glob("batch_*.jsonl"))
YOUTUBE_FILES = sorted((ROOT / "Youtube_scrap" / "data" / "processed").glob("batch_*.jsonl"))

DJELFA_TARGET = 1_500
YOUTUBE_BUCKET_TARGETS = {"arabic": 2_834, "latin": 2_833, "mixed": 2_833}
OVERSAMPLE_FACTOR = 1.3
SEED = 42

# Same regex convention reused verbatim across this repo (N-gram/,
# Youtube_scrap/, Tokenization/) for script classification.
ARABIC_RE = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")
LATIN_RE = re.compile(r"[a-zA-ZÀ-ɏ]")

NUM_PERM = 128
SHINGLE_SIZE = 4
DEDUP_THRESHOLD = 0.8
_WHITESPACE_RE = re.compile(r"\s+")


def script_of(text: str) -> str:
    has_ar = bool(ARABIC_RE.search(text))
    has_lat = bool(LATIN_RE.search(text))
    if has_ar and has_lat:
        return "mixed"
    if has_ar:
        return "arabic"
    if has_lat:
        return "latin"
    return "other"


# Derived and validated in notebooks/01_cleaning_rules.ipynb against 200
# real samples -- strips the [MENTION]/[URL] anonymization placeholders
# and, if present, the glued-on reply-ID fragment right after [MENTION]
# (bounded to 1-10 chars so it can never eat into a real word). Used only
# to decide the script bucket below; the stored row keeps the original text.
_MENTION_WITH_FRAGMENT_RE = re.compile(r"\[MENTION\](\s*-[^\s]{1,10})?")
_URL_RE = re.compile(r"\[URL\]")
_INLINE_WHITESPACE_RE = re.compile(r"[ \t]+")


def clean_for_classification(text: str) -> str:
    text = _MENTION_WITH_FRAGMENT_RE.sub("", text)
    text = _URL_RE.sub("", text)
    return _INLINE_WHITESPACE_RE.sub(" ", text).strip()


def _shingles(text: str, k: int = SHINGLE_SIZE) -> set[str]:
    normalized = _WHITESPACE_RE.sub(" ", text.strip().lower())
    if not normalized:
        return set()
    if len(normalized) <= k:
        return {normalized}
    return {normalized[i : i + k] for i in range(len(normalized) - k + 1)}


def _minhash(text: str) -> MinHash:
    mh = MinHash(num_perm=NUM_PERM)
    for shingle in _shingles(text):
        mh.update(shingle.encode("utf-8"))
    return mh


def dedup_and_trim(rows: list[dict], target: int, label: str) -> list[dict]:
    """Walks `rows` in (already-shuffled) order, drops near-duplicates via
    a fresh MinHash/LSH index, stops once `target` unique rows are kept.
    """
    lsh = MinHashLSH(threshold=DEDUP_THRESHOLD, num_perm=NUM_PERM)
    kept: list[dict] = []
    for i, row in enumerate(rows):
        if len(kept) >= target:
            break
        mh = _minhash(row["text"])
        if lsh.query(mh):
            continue
        lsh.insert(row["id"], mh)
        kept.append(row)
    print(f"  {label}: {len(kept)}/{target} after dedup (from {min(i + 1, len(rows))} candidates scanned)")
    if len(kept) < target:
        print(f"  WARNING: {label} came up short -- raise OVERSAMPLE_FACTOR and rerun")
    return kept


def reservoir_sample_djelfa(target_pool: int, rng: random.Random) -> list[dict]:
    reservoir: list[dict] = []
    seen = 0
    for path in DJELFA_FILES:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not row.get("text", "").strip():
                    continue
                seen += 1
                if len(reservoir) < target_pool:
                    reservoir.append(row)
                else:
                    j = rng.randint(0, seen - 1)
                    if j < target_pool:
                        reservoir[j] = row
    print(f"djelfa: reservoir-sampled {len(reservoir)} from {seen:,} candidates")
    return reservoir


def reservoir_sample_youtube_buckets(target_pools: dict[str, int], rng: random.Random) -> dict[str, list[dict]]:
    reservoirs: dict[str, list[dict]] = {k: [] for k in target_pools}
    seen: dict[str, int] = {k: 0 for k in target_pools}
    for i, path in enumerate(YOUTUBE_FILES):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                text = row.get("text", "")
                if not text.strip():
                    continue
                bucket = script_of(clean_for_classification(text))
                if bucket not in reservoirs:
                    continue  # "other" -- no real script content, skip
                seen[bucket] += 1
                target = target_pools[bucket]
                res = reservoirs[bucket]
                if len(res) < target:
                    res.append(row)
                else:
                    j = rng.randint(0, seen[bucket] - 1)
                    if j < target:
                        res[j] = row
        if (i + 1) % 5 == 0:
            print(f"  ...{i + 1}/{len(YOUTUBE_FILES)} YouTube batch files scanned")
    for bucket, res in reservoirs.items():
        print(f"youtube[{bucket}]: reservoir-sampled {len(res)} from {seen[bucket]:,} candidates")
    return reservoirs


def main() -> None:
    rng = random.Random(SEED)

    djelfa_pool = reservoir_sample_djelfa(int(DJELFA_TARGET * OVERSAMPLE_FACTOR), rng)
    rng.shuffle(djelfa_pool)
    djelfa_final = dedup_and_trim(djelfa_pool, DJELFA_TARGET, "djelfa")
    for row in djelfa_final:
        row["sample_group"] = "djelfa"

    youtube_pools = reservoir_sample_youtube_buckets(
        {k: int(v * OVERSAMPLE_FACTOR) for k, v in YOUTUBE_BUCKET_TARGETS.items()}, rng
    )
    youtube_final: list[dict] = []
    for bucket, target in YOUTUBE_BUCKET_TARGETS.items():
        pool = youtube_pools[bucket]
        rng.shuffle(pool)
        rows = dedup_and_trim(pool, target, f"youtube[{bucket}]")
        for row in rows:
            row["sample_group"] = f"youtube_{bucket}"
        youtube_final.extend(rows)

    all_rows = djelfa_final + youtube_final
    rng.shuffle(all_rows)  # mix sample_groups together so annotation isn't biased by batch order

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps({"id": row["id"], "text": row["text"], "sample_group": row["sample_group"]}, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(all_rows):,} rows to {OUT_PATH}")
    from collections import Counter
    print("By sample_group:", dict(Counter(r["sample_group"] for r in all_rows)))


if __name__ == "__main__":
    main()
