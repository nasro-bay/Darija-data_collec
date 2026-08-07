#!/usr/bin/env python
"""CLI: run the near-dup dedup pipeline over newly scraped raw posts,
writing data/processed/batch_<date>.jsonl and appending a run entry
(with a per-subforum breakdown) to data/logs/log.json.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from darija_forum.pipeline import run_pipeline  # noqa: E402
from darija_forum.state import State  # noqa: E402


def main() -> None:
    state = State(ROOT / "data" / "state" / "crawl_state.json")
    result = run_pipeline(
        raw_dir=ROOT / "data" / "raw" / "djelfa",
        processed_dir=ROOT / "data" / "processed",
        state=state,
        lsh_path=ROOT / "data" / "state" / "minhash_lsh.pkl",
        log_path=ROOT / "data" / "logs" / "log.json",
    )
    state.save()
    print(
        f"Threads processed: {result['threads_processed']}  "
        f"Posts collected: {result['posts_collected']}  "
        f"Retained after dedup: {result['posts_retained']}"
    )
    if result["by_subforum"]:
        print("\nBy subforum:")
        for subforum_id, counts in result["by_subforum"].items():
            print(f"  f={subforum_id}: {counts['posts_retained']}/{counts['posts_collected']} retained")


if __name__ == "__main__":
    main()
