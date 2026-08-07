#!/usr/bin/env python
"""CLI: run the near-dup dedup pipeline over newly scraped raw comments,
writing data/processed/batch_<date>.jsonl and appending a run entry to
data/logs/log.json.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from darija_corpus.pipeline import run_pipeline  # noqa: E402
from darija_corpus.state import State  # noqa: E402


def main() -> None:
    state = State(ROOT / "data" / "state" / "scrape_state.json")
    result = run_pipeline(
        raw_dir=ROOT / "data" / "raw",
        processed_dir=ROOT / "data" / "processed",
        state=state,
        lsh_path=ROOT / "data" / "state" / "minhash_lsh.pkl",
        log_path=ROOT / "data" / "logs" / "log.json",
    )
    state.save()
    print(
        f"Raw files processed: {result['videos_scraped']}  "
        f"Comments collected: {result['comments_collected']}  "
        f"Retained after dedup: {result['comments_retained']}"
    )


if __name__ == "__main__":
    main()
