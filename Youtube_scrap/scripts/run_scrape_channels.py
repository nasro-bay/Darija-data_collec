#!/usr/bin/env python
"""CLI: scrape raw comments for the channels configured in
config/seed_channels.yaml, newest videos first. Resumable: safe to re-run
after a crash or after the daily quota is exhausted; a channel already
fully walked (or capped by `max_videos`) is skipped on rerun.
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

from darija_corpus.scrape import extract_channel_ref, scrape_channel  # noqa: E402
from darija_corpus.state import QuotaExceededError, State  # noqa: E402
from darija_corpus.youtube_client import YouTubeClient  # noqa: E402


def load_seed_channels(config_path: Path) -> list[dict]:
    """Returns a normalized list of {"channel_id"|"handle", "max_videos"} dicts.

    Entries in the YAML may be a dict (`channel_id:`/`handle:`/`max_videos:`)
    or a plain string — a channel URL, a bare @handle, or a bare channel ID.
    """
    if not config_path.exists():
        raise SystemExit(f"Seed channel config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    raw_entries = data.get("channels") or []

    normalized = []
    for entry in raw_entries:
        if isinstance(entry, str):
            ref = extract_channel_ref(entry)
            normalized.append({**ref, "max_videos": None})
        else:
            ref = {"channel_id": entry.get("channel_id")} if entry.get("channel_id") else (
                {"handle": entry.get("handle")} if entry.get("handle") else {}
            )
            normalized.append({**ref, "max_videos": entry.get("max_videos")})
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config" / "seed_channels.yaml"))
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Cap every channel in this run to at most N newest videos, "
        "overriding a channel's own max_videos only if it's stricter (lower). "
        "Handy for a cheap test run without editing the config.",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="parallel worker processes per page of videos, each scraping a "
        "different video's comments (default: CPU count)",
    )
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        raise SystemExit("YOUTUBE_API_KEY not set. Copy .env.example to .env and fill it in.")

    channels = load_seed_channels(Path(args.config))
    if not channels:
        raise SystemExit(
            f"No channels configured in {args.config}. "
            "Add at least one channel entry under `channels:` "
            "(see scripts/discover_channels.py to help find real ones)."
        )

    state = State(ROOT / "data" / "state" / "scrape_state.json")
    client = YouTubeClient(api_key, state)
    raw_dir = ROOT / "data" / "raw"

    for entry in channels:
        channel_id = entry.get("channel_id")
        handle = entry.get("handle")
        entry_max = entry.get("max_videos")
        max_videos = min(v for v in (entry_max, args.max_videos) if v is not None) if (
            entry_max is not None or args.max_videos is not None
        ) else None
        label = channel_id or handle
        if not label:
            print(f"Skipping unrecognized channel entry: {entry}")
            continue
        print(f"Scraping channel: {label}" + (f" (max_videos={max_videos})" if max_videos else " (NO CAP — will walk the whole channel)"))
        try:
            result = scrape_channel(
                client, state, raw_dir,
                channel_id=channel_id, handle=handle, max_videos=max_videos,
                api_key=api_key, workers=args.workers,
            )
            print(f"  -> {result}")
            if result.get("quota_exceeded"):
                print("  Stopped: daily quota exhausted.")
                print("  Progress saved — rerun this script (tomorrow, once quota resets) to resume.")
                break
        except QuotaExceededError as exc:
            print(f"  Stopped: {exc}")
            print("  Progress saved — rerun this script (tomorrow, once quota resets) to resume.")
            break

    print(f"\nQuota used today: {state.data['quota']['units_used']} / 10000")


if __name__ == "__main__":
    main()
