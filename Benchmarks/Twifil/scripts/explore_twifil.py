#!/usr/bin/env python
"""Downloads arbml/Twifil (via `datasets`) and computes the stats behind
README.md's "What's actually in it" section: field distributions, script
composition (raw vs. mention/URL-stripped -- the two differ a lot, see
README), duplicate/near-empty rates, and the tweet-ID float-precision
issue. Prints a report and writes two artifacts to ../data/ (gitignored):

  - twifil_cleaned.jsonl -- one row per tweet, `Post` with URLs/mentions
    replaced by [URL]/[MENTION] placeholders (same convention as
    Youtube_scrap/Mountada_djelfa_scrap's clean_text.py), plus the label
    fields likely to matter for downstream use (lang, Polarity Class,
    Emotion). Not full cleaning (no dedup, no elongation/punctuation
    collapsing) -- just enough to make the script-distribution numbers
    trustworthy and give downstream scripts a lighter file to load.
  - twifil_report.json -- every number in this script's printed report,
    for reuse without re-downloading/re-computing.

Rerun if the upstream dataset changes.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]  # Benchmarks/Twifil/
DATA_DIR = ROOT / "data"

ARABIC_RE = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")
LATIN_RE = re.compile(r"[a-zA-ZÀ-ɏ]")
URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@\w+")


def strip_noise(text: str) -> str:
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    return text.strip()


def placeholder_text(text: str) -> str:
    text = URL_RE.sub("[URL]", text)
    text = MENTION_RE.sub("[MENTION]", text)
    return text


def classify_script(text: str) -> str:
    has_ar = bool(ARABIC_RE.search(text))
    has_lat = bool(LATIN_RE.search(text))
    if has_ar and has_lat:
        return "mixed"
    if has_ar:
        return "arabic"
    if has_lat:
        return "latin"
    return "other_empty"


def main() -> None:
    ds = load_dataset("arbml/Twifil")["train"]
    posts: list[str] = ds["Post"]
    lang: list[str] = ds["lang"]

    report: dict = {"num_rows": len(ds), "columns": ds.column_names}

    # --- ID / Code field sanity ---
    report["id_unique_count"] = len(set(ds["ID"]))
    report["code_sample"] = ds["Code"][:3]  # scientific-notation strings -- see README

    # --- missing / duplicate / length ---
    report["empty_post_count"] = sum(1 for p in posts if not p.strip())
    post_counts = Counter(posts)
    report["duplicate_post_rows"] = sum(c for c in post_counts.values() if c > 1)
    report["unique_post_count"] = len(post_counts)
    lens = sorted(len(p) for p in posts)
    n = len(lens)
    report["post_length_chars"] = {
        "min": lens[0],
        "max": lens[-1],
        "median": lens[n // 2],
        "mean": round(sum(lens) / n, 2),
    }

    # --- categorical field distributions ---
    for field in ("lang", "Polarity Class", "Platform", "Profile Lang"):
        report[f"{field}_distribution"] = dict(Counter(ds[field]).most_common(15))
    emotion_counts = Counter(ds["Emotion"])
    report["emotion_non_nan_count"] = sum(c for v, c in emotion_counts.items() if v != "nan")
    report["emotion_distribution_non_nan"] = {
        v: c for v, c in emotion_counts.most_common(20) if v != "nan"
    }

    # --- script distribution: raw vs. mention/URL-stripped ---
    raw_script = [classify_script(p) for p in posts]
    report["script_distribution_raw"] = dict(Counter(raw_script))

    cleaned_for_script = [strip_noise(p) for p in posts]
    stripped_script = [classify_script(c) for c in cleaned_for_script]
    report["script_distribution_stripped"] = dict(Counter(stripped_script))
    report["empty_after_stripping_mentions_urls"] = sum(1 for c in cleaned_for_script if not c)

    # cross-tab: lang field vs. actual (stripped) script, for the 4 main lang values
    cross = Counter(zip(lang, stripped_script))
    report["lang_vs_stripped_script"] = {
        l: {s: cross.get((l, s), 0) for s in ("arabic", "latin", "mixed", "other_empty")}
        for l in ("ar", "fr", "en", "und")
    }

    # --- print report ---
    print(json.dumps(report, ensure_ascii=False, indent=2))

    # --- write artifacts ---
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "twifil_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with (DATA_DIR / "twifil_cleaned.jsonl").open("w", encoding="utf-8") as f:
        for i in range(len(ds)):
            row = {
                "id": ds["ID"][i],
                "text": placeholder_text(posts[i]),
                "lang": ds["lang"][i],
                "polarity_class": ds["Polarity Class"][i],
                "emotion": ds["Emotion"][i] if ds["Emotion"][i] != "nan" else None,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nWrote {DATA_DIR / 'twifil_report.json'}")
    print(f"Wrote {DATA_DIR / 'twifil_cleaned.jsonl'} ({len(ds)} rows)")


if __name__ == "__main__":
    main()
