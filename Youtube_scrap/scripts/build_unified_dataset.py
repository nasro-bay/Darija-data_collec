#!/usr/bin/env python
"""Creates a unified JSONL dataset under Data/ containing only `id` and `text`
from all processed YouTube data batches (Youtube_scrap/data/processed/batch_*.jsonl).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = ROOT / "Youtube_scrap" / "data" / "processed"
OUTPUT_PATH = ROOT / "Data" / "youtube_corpus.jsonl"


def main() -> None:
    if not PROCESSED_DIR.exists():
        raise SystemExit(f"Processed directory not found: {PROCESSED_DIR}")

    batch_files = sorted(list(PROCESSED_DIR.glob("batch_*.jsonl")))
    if not batch_files:
        raise SystemExit(f"No batch files found in {PROCESSED_DIR}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Reading from {len(batch_files)} batch files...")

    total_count = 0
    with OUTPUT_PATH.open("w", encoding="utf-8") as out_f:
        for batch_file in batch_files:
            print(f" Processing {batch_file.name}...")
            with batch_file.open("r", encoding="utf-8") as in_f:
                for line in in_f:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        doc_id = record.get("id")
                        text = record.get("text")
                        if doc_id and text is not None:
                            unified = {"id": doc_id, "text": text}
                            out_f.write(json.dumps(unified, ensure_ascii=False) + "\n")
                            total_count += 1
                    except Exception as e:
                        print(f" Error parsing line: {e}")

    print(f"Done! Unified dataset written to: {OUTPUT_PATH}")
    print(f"Total documents: {total_count:,}")


if __name__ == "__main__":
    main()
