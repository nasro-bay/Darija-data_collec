#!/usr/bin/env python
"""CLI: counts how many comments were scraped *today*, straight from the
raw JSONL files -- no need to run run_pipeline.py first.

Works because every comment record is stamped with `scrape_date` (the
date it was actually scraped) at scrape time, in scrape.py's
_comment_record() -- independent of which raw file it ends up in, and
independent of run_pipeline's own counters (which only exist after a
pipeline run, and even then log.json's cumulative totals have a known
reliability gap -- see CLAUDE.md's note on the 2026-08-14 zero-stats
bug). Counting straight from `scrape_date` sidesteps both.

    python scripts/count_today_scraped.py
    python scripts/count_today_scraped.py --date 2026-08-30
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--date", default=None,
        help="date to count (YYYY-MM-DD), defaults to today",
    )
    args = parser.parse_args()
    target_date = args.date or date.today().isoformat()

    raw_dir = ROOT / "data" / "raw"
    count = 0
    files_touched = 0
    for path in sorted(raw_dir.glob("*.jsonl")):
        file_count = 0
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("scrape_date") == target_date:
                    file_count += 1
        if file_count:
            files_touched += 1
            count += file_count

    print(f"{target_date}: {count:,} comments scraped, across {files_touched:,} raw video file(s)")


if __name__ == "__main__":
    main()
