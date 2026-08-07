#!/usr/bin/env python
"""CLI: scrape raw comments for the video links configured in
config/seed_videos.yaml. Resumable: safe to re-run after a crash or
after the daily quota is exhausted.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from darija_corpus.scrape import scrape_video  # noqa: E402
from darija_corpus.state import QuotaExceededError, State  # noqa: E402
from darija_corpus.youtube_client import YouTubeClient  # noqa: E402


def load_seed_videos(config_path: Path) -> list[str]:
    if not config_path.exists():
        raise SystemExit(f"Seed video config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("videos") or []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config" / "seed_videos.yaml"))
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        raise SystemExit("YOUTUBE_API_KEY not set. Copy .env.example to .env and fill it in.")

    videos = load_seed_videos(Path(args.config))
    if not videos:
        raise SystemExit(
            f"No videos configured in {args.config}. "
            "Add at least one video URL (or bare video ID) under `videos:`."
        )

    state = State(ROOT / "data" / "state" / "scrape_state.json")
    client = YouTubeClient(api_key, state)
    raw_dir = ROOT / "data" / "raw"

    for entry in videos:
        print(f"Scraping video: {entry}")
        try:
            result = scrape_video(client, state, raw_dir, entry)
            print(f"  -> {result}")
        except QuotaExceededError as exc:
            print(f"  Stopped: {exc}")
            print("  Progress saved — rerun this script (tomorrow, once quota resets) to resume.")
            break

    print(f"\nQuota used today: {state.data['quota']['units_used']} / 10000")


if __name__ == "__main__":
    main()
