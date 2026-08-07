#!/usr/bin/env python
"""CLI: discover candidate Algerian YouTube channels worth browsing for
video links.

Prints candidates for manual review — use the channel_id to open the
channel, then paste specific video links you want into
config/seed_videos.yaml.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

from darija_corpus.discover import discover_channels  # noqa: E402
from darija_corpus.state import State  # noqa: E402
from darija_corpus.youtube_client import YouTubeClient  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keywords", help="Comma-separated search keywords (overrides defaults)")
    parser.add_argument("--max-results", type=int, default=10)
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        raise SystemExit("YOUTUBE_API_KEY not set. Copy .env.example to .env and fill it in.")

    state = State(ROOT / "data" / "state" / "scrape_state.json")
    client = YouTubeClient(api_key, state)

    keywords = [k.strip() for k in args.keywords.split(",")] if args.keywords else None
    candidates = discover_channels(client, keywords=keywords, max_results_per_keyword=args.max_results)
    state.save()

    print(f"Found {len(candidates)} candidate channel(s):\n")
    for c in candidates:
        print(f"- {c['title']}  (channel_id: {c['channel_id']})")
        print(f"  matched keyword: {c['matched_keyword']!r}")
        if c["description"]:
            print(f"  {c['description'][:120]}")
        print()
    print("Open a channel that fits, grab video links from it, and add them to config/seed_videos.yaml")
    print(f"Quota used today: {state.data['quota']['units_used']} / 10000")


if __name__ == "__main__":
    main()
