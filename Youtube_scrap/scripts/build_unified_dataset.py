#!/usr/bin/env python
"""Builds Data/youtube_corpus.jsonl (id + text only) from all processed
YouTube batches, then shuffles it into Data/youtube_corpus_shuffled.jsonl
(external chunk-shuffle -- doesn't need the whole corpus in RAM), then
copies that shuffled file into the DarijaDZ/ and Kaggle_DarijaDz/ release
folders (as youtube_corpus.jsonl -- the shuffled version is what actually
gets published, not the raw build order), then updates the numeric
statistics already present in the DarijaDZ/, Kaggle_DarijaDz/, and
DarijaDz_Tokenizers/ dataset/model-card README.md files to match.

One command now does what used to be two separate scripts
(build_unified_dataset.py + shuffle_unified_dataset.py, the latter now
removed) -- shuffling always happens as part of the build.

README numbers are updated in place via targeted regex substitution keyed
to each row/line's existing label text (e.g. "Documents", "Arabic
script", "Channels:") -- only the numeric values change, nothing else
about the files' wording or structure. If a README's wording around a
stat changes later, the matching substitution here will silently stop
firing (a warning is printed) rather than corrupt the file.
"""
from __future__ import annotations

import json
import random
import re
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = ROOT / "Youtube_scrap" / "data" / "processed"
UNIFIED_PATH = ROOT / "Data" / "youtube_corpus.jsonl"
SHUFFLED_PATH = ROOT / "Data" / "youtube_corpus_shuffled.jsonl"
README_PATHS = [ROOT / "DarijaDZ" / "README.md", ROOT / "Kaggle_DarijaDz" / "README.md"]
TOKENIZER_README_PATH = ROOT / "DarijaDz_Tokenizers" / "README.md"
RELEASE_CORPUS_PATHS = [ROOT / "DarijaDZ" / "youtube_corpus.jsonl", ROOT / "Kaggle_DarijaDz" / "youtube_corpus.jsonl"]

SHUFFLE_SEED = 42
SHUFFLE_CHUNK_SIZE = 100_000

# Same script-classification rule used in Notebooks/04_youtube_corpus_stats.ipynb --
# kept in sync with it so the README's script-distribution numbers match
# what that notebook would report over the same corpus.
ARABIC_RE = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻾]")
LATIN_RE = re.compile(r"[a-zA-ZÀ-ɏ]")


def classify_token(token: str) -> str:
    has_arabic = bool(ARABIC_RE.search(token))
    has_latin = bool(LATIN_RE.search(token))
    if has_arabic and not has_latin:
        return "arabic"
    if has_latin and not has_arabic:
        return "latin"
    if has_arabic and has_latin:
        return "mixed"
    return "digits_symbols"


def build_unified(batch_files: list[Path]) -> dict:
    """Streams every batch file once: writes the unified id+text JSONL and
    accumulates the stats needed for the README updates (token/script
    counts, distinct channels, distinct video files) along the way, since
    the full per-document records (with `channel`/`video_id`) are only
    available here -- the unified file itself drops everything but
    `id`/`text`.
    """
    UNIFIED_PATH.parent.mkdir(parents=True, exist_ok=True)

    total_docs = 0
    total_tokens = 0
    script_token_counts: Counter[str] = Counter()
    channels: set[str] = set()
    video_ids: set[str] = set()

    print(f"Reading from {len(batch_files)} batch files...")
    with UNIFIED_PATH.open("w", encoding="utf-8") as out_f:
        for batch_file in batch_files:
            print(f"  Processing {batch_file.name}...")
            with batch_file.open("r", encoding="utf-8") as in_f:
                for line in in_f:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"  Error parsing line: {e}")
                        continue

                    doc_id = record.get("id")
                    text = record.get("text")
                    if not doc_id or text is None:
                        continue

                    out_f.write(json.dumps({"id": doc_id, "text": text}, ensure_ascii=False) + "\n")
                    total_docs += 1

                    tokens = text.split()
                    total_tokens += len(tokens)
                    for tok in tokens:
                        script_token_counts[classify_token(tok)] += 1

                    channel = record.get("channel")
                    if channel:
                        channels.add(channel)
                    video_id = record.get("video_id")
                    if video_id:
                        video_ids.add(video_id)

    print(f"Done! Unified dataset written to: {UNIFIED_PATH}")
    print(f"Total documents: {total_docs:,}")

    return {
        "total_docs": total_docs,
        "total_tokens": total_tokens,
        "script_token_counts": script_token_counts,
        "channels": len(channels),
        "video_files": len(video_ids),
    }


def shuffle_unified() -> None:
    """External chunk-shuffle of UNIFIED_PATH -> SHUFFLED_PATH: shuffle
    within fixed-size chunks, shuffle chunk order, concatenate. Same
    algorithm previously in the now-removed shuffle_unified_dataset.py --
    avoids loading the multi-million-line file into RAM at once.
    """
    rng = random.Random(SHUFFLE_SEED)
    temp_dir = ROOT / "Data" / "_shuffle_chunks"
    temp_dir.mkdir(parents=True, exist_ok=True)

    print("\nShuffling unified dataset...")
    chunk: list[dict] = []
    chunk_files: list[Path] = []
    chunk_number = 0
    total_count = 0

    def _flush_chunk() -> None:
        nonlocal chunk, chunk_number
        rng.shuffle(chunk)
        chunk_path = temp_dir / f"chunk_{chunk_number:05d}.jsonl"
        with chunk_path.open("w", encoding="utf-8") as out_f:
            for record in chunk:
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        chunk_files.append(chunk_path)
        chunk = []
        chunk_number += 1

    with UNIFIED_PATH.open("r", encoding="utf-8") as in_f:
        for line in in_f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  Error parsing line: {e}")
                continue
            chunk.append(record)
            total_count += 1
            if len(chunk) >= SHUFFLE_CHUNK_SIZE:
                _flush_chunk()
    if chunk:
        _flush_chunk()

    print(f"  Created {len(chunk_files)} chunks ({total_count:,} documents).")
    rng.shuffle(chunk_files)

    print("Writing final shuffled dataset...")
    with SHUFFLED_PATH.open("w", encoding="utf-8") as out_f:
        for i, chunk_path in enumerate(chunk_files, start=1):
            print(f"  Merging chunk {i}/{len(chunk_files)}...")
            with chunk_path.open("r", encoding="utf-8") as chunk_f:
                for line in chunk_f:
                    out_f.write(line)
            chunk_path.unlink()
    temp_dir.rmdir()

    print(f"Done! Shuffled dataset written to: {SHUFFLED_PATH}")
    print(f"Total documents: {total_count:,}")


def sync_release_copies() -> None:
    """Copies the shuffled corpus into the DarijaDZ/ and Kaggle_DarijaDz/
    release folders as youtube_corpus.jsonl -- those folders previously
    went stale relative to Data/ because nothing re-synced them after a
    rebuild (caught manually once; this closes that gap for good). Always
    overwrites from scratch, same as every other output this script
    produces.
    """
    for dest in RELEASE_CORPUS_PATHS:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SHUFFLED_PATH, dest)
        print(f"  Synced {dest.relative_to(ROOT)}")


def _sub_bold_number(text: str, label_pattern: str, value: str) -> str:
    """Replaces the bold `**number[%]**` that follows `label_pattern`
    (matched within the same line only -- `.` doesn't span newlines here),
    leaving everything else in the line untouched.
    """
    pattern = re.compile(rf"({label_pattern}.*?\*\*)[\d,.]+%?(\*\*)")
    new_text, n = pattern.subn(lambda m: m.group(1) + value + m.group(2), text, count=1)
    if n == 0:
        print(f"  WARNING: pattern not found for {label_pattern!r} -- a README line was left unchanged")
    return new_text


def _sub_plain_number(text: str, label: str, value: str) -> str:
    """Replaces a plain (non-bold) `| number |` table cell following a
    `| <label>` row label.
    """
    pattern = re.compile(rf"(\|\s*{label}\s*\|\s*)[\d,]+(\s*\|)")
    new_text, n = pattern.subn(lambda m: m.group(1) + value + m.group(2), text, count=1)
    if n == 0:
        print(f"  WARNING: pattern not found for {label!r} -- a README line was left unchanged")
    return new_text


def _sub_intro_prose(text: str, total_docs: int, total_tokens: int) -> str:
    """Replaces the rounded "approximately **X.X million comment
    documents** and **Y.YY million word-level tokens**" intro sentence.
    Distinct from _sub_bold_number because the bold spans here wrap the
    whole "N million <unit>" phrase, not just the number -- the closing
    ** doesn't immediately follow the digits like it does in the stats
    table, so _sub_bold_number's assumption doesn't hold here.
    """
    pattern = re.compile(
        r"approximately \*\*[\d.]+ million comment documents\*\* "
        r"and \*\*[\d.]+ million word-level tokens\*\*"
    )
    replacement = (
        f"approximately **{total_docs / 1_000_000:.1f} million comment documents** "
        f"and **{total_tokens / 1_000_000:.2f} million word-level tokens**"
    )
    new_text, n = pattern.subn(replacement, text, count=1)
    if n == 0:
        print(
            "  WARNING: pattern not found for the intro 'approximately X million...' "
            "sentence -- a README line was left unchanged"
        )
    return new_text


def update_tokenizer_readme(stats: dict) -> None:
    """DarijaDz_Tokenizers/README.md's "Training data" section states the
    YouTube corpus size as a rounded "~X.XXM Algerian YouTube comments"
    figure (not a table cell like the other READMEs, so it needs its own
    substitution pattern). Only that number is touched -- the adjacent
    "~1.17M documents used for training" figure is a *combined*
    YouTube+djelfa training-corpus count (Tokenization/data/train_corpus.txt),
    which this script has no visibility into (it only reads YouTube's
    processed batches) and shouldn't guess at.
    """
    if not TOKENIZER_README_PATH.exists():
        print(f"  WARNING: {TOKENIZER_README_PATH} not found, skipping")
        return

    millions = stats["total_docs"] / 1_000_000
    text = TOKENIZER_README_PATH.read_text(encoding="utf-8")
    pattern = re.compile(r"~[\d.]+M Algerian YouTube comments")
    new_text, n = pattern.subn(f"~{millions:.2f}M Algerian YouTube comments", text, count=1)
    if n == 0:
        print(
            "  WARNING: pattern not found for '~X.XXM Algerian YouTube comments' "
            "-- DarijaDz_Tokenizers/README.md left unchanged"
        )
        return

    TOKENIZER_README_PATH.write_text(new_text, encoding="utf-8")
    print(f"  Updated {TOKENIZER_README_PATH.relative_to(ROOT)}")


def update_readmes(stats: dict) -> None:
    total_docs = stats["total_docs"]
    total_tokens = stats["total_tokens"]
    mean_tokens = total_tokens / total_docs if total_docs else 0.0
    counts = stats["script_token_counts"]
    arabic = counts.get("arabic", 0)
    latin = counts.get("latin", 0)
    mixed = counts.get("mixed", 0)
    digits = counts.get("digits_symbols", 0)

    def pct(n: int) -> str:
        return f"{(n / total_tokens * 100):.2f}%" if total_tokens else "0.00%"

    for readme_path in README_PATHS:
        if not readme_path.exists():
            print(f"  WARNING: {readme_path} not found, skipping")
            continue
        text = readme_path.read_text(encoding="utf-8")

        text = _sub_intro_prose(text, total_docs, total_tokens)

        text = _sub_bold_number(text, r"\|\s*Documents\s*\|", f"{total_docs:,}")
        text = _sub_bold_number(text, r"\|\s*Word-level tokens\s*\|", f"{total_tokens:,}")
        text = _sub_bold_number(text, r"\|\s*Mean tokens / document\s*\|", f"{mean_tokens:.2f}")

        text = _sub_plain_number(text, "Arabic script", f"{arabic:,}")
        text = _sub_bold_number(text, r"\|\s*Arabic script\s*\|\s*[\d,]+\s*\|", pct(arabic))

        text = _sub_plain_number(text, "Latin / Arabizi", f"{latin:,}")
        text = _sub_bold_number(text, r"\|\s*Latin / Arabizi\s*\|\s*[\d,]+\s*\|", pct(latin))

        text = _sub_plain_number(text, r"Digits, punctuation & symbols", f"{digits:,}")
        text = _sub_bold_number(text, r"\|\s*Digits, punctuation & symbols\s*\|\s*[\d,]+\s*\|", pct(digits))

        text = _sub_plain_number(text, r"Mixed Arabic \+ Latin", f"{mixed:,}")
        text = _sub_bold_number(text, r"\|\s*Mixed Arabic \+ Latin\s*\|\s*[\d,]+\s*\|", pct(mixed))

        text = _sub_bold_number(text, r"\*\s*Channels:", f"{stats['channels']}")
        text = _sub_bold_number(text, r"\*\s*Raw video files processed:", f"{stats['video_files']:,}")

        readme_path.write_text(text, encoding="utf-8")
        print(f"  Updated {readme_path.relative_to(ROOT)}")


def main() -> None:
    if not PROCESSED_DIR.exists():
        raise SystemExit(f"Processed directory not found: {PROCESSED_DIR}")

    batch_files = sorted(PROCESSED_DIR.glob("batch_*.jsonl"))
    if not batch_files:
        raise SystemExit(f"No batch files found in {PROCESSED_DIR}")

    stats = build_unified(batch_files)
    shuffle_unified()

    print("\nSyncing release copies...")
    sync_release_copies()

    print("\nUpdating README statistics...")
    update_readmes(stats)
    update_tokenizer_readme(stats)


if __name__ == "__main__":
    main()
