#!/usr/bin/env python
"""Shuffle a JSONL dataset without loading the entire dataset into RAM."""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_PATH = ROOT / "Data" / "youtube_corpus.jsonl"
OUTPUT_PATH = ROOT / "Data" / "youtube_corpus_shuffled.jsonl"

SEED = 42
CHUNK_SIZE = 100_000


def main() -> None:
    if not INPUT_PATH.exists():
        raise SystemExit(f"Input file not found: {INPUT_PATH}")

    rng = random.Random(SEED)

    print(f"Input:  {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Seed:   {SEED}")
    print()

    # Temporary chunks
    temp_dir = ROOT / "Data" / "_shuffle_chunks"
    temp_dir.mkdir(parents=True, exist_ok=True)

    chunk = []
    chunk_files = []
    total_count = 0
    chunk_number = 0

    # ---------------------------------------------------------
    # Step 1: Create shuffled chunks
    # ---------------------------------------------------------
    print("Creating shuffled chunks...")

    with INPUT_PATH.open("r", encoding="utf-8") as in_f:

        for line in in_f:
            if not line.strip():
                continue

            try:
                record = json.loads(line)

                if record.get("id") and record.get("text") is not None:
                    chunk.append(record)
                    total_count += 1

                if len(chunk) >= CHUNK_SIZE:
                    rng.shuffle(chunk)

                    chunk_path = temp_dir / f"chunk_{chunk_number:05d}.jsonl"

                    with chunk_path.open("w", encoding="utf-8") as out_f:
                        for record in chunk:
                            out_f.write(
                                json.dumps(
                                    record,
                                    ensure_ascii=False
                                ) + "\n"
                            )

                    chunk_files.append(chunk_path)

                    print(
                        f"  Created chunk {chunk_number}: "
                        f"{len(chunk):,} documents"
                    )

                    chunk.clear()
                    chunk_number += 1

            except json.JSONDecodeError as e:
                print(f"Error parsing line: {e}")

    # Write remaining records
    if chunk:
        rng.shuffle(chunk)

        chunk_path = temp_dir / f"chunk_{chunk_number:05d}.jsonl"

        with chunk_path.open("w", encoding="utf-8") as out_f:
            for record in chunk:
                out_f.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

        chunk_files.append(chunk_path)

    print()
    print(f"Total documents: {total_count:,}")
    print(f"Created {len(chunk_files)} chunks.")
    print()

    # ---------------------------------------------------------
    # Step 2: Shuffle chunk order
    # ---------------------------------------------------------
    rng.shuffle(chunk_files)

    # ---------------------------------------------------------
    # Step 3: Merge shuffled chunks
    # ---------------------------------------------------------
    print("Writing final shuffled dataset...")

    with OUTPUT_PATH.open("w", encoding="utf-8") as out_f:

        for i, chunk_path in enumerate(chunk_files, start=1):

            print(
                f"  Merging chunk {i}/{len(chunk_files)}..."
            )

            with chunk_path.open("r", encoding="utf-8") as chunk_f:
                for line in chunk_f:
                    out_f.write(line)

    # ---------------------------------------------------------
    # Step 4: Remove temporary chunks
    # ---------------------------------------------------------
    print()
    print("Removing temporary files...")

    for chunk_path in chunk_files:
        chunk_path.unlink()

    temp_dir.rmdir()

    print()
    print("Done!")
    print(f"Shuffled dataset: {OUTPUT_PATH}")
    print(f"Total documents: {total_count:,}")


if __name__ == "__main__":
    main()