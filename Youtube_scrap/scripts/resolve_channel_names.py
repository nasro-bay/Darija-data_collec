#!/usr/bin/env python
"""Resolves channel IDs from processed data and scrape state into channel titles/handles,
saving the mapping to data/state/channel_names.json.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
from darija_corpus.state import State
from darija_corpus.youtube_client import YouTubeClient


def collect_unique_channel_ids(root: Path, state: State) -> set[str]:
    channel_ids = set()

    # From scrape_state.json
    for v_info in state.data.get("videos", {}).values():
        c_id = v_info.get("channel_id")
        if c_id:
            channel_ids.add(c_id)

    for c_id in state.data.get("channels", {}).keys():
        if c_id:
            channel_ids.add(c_id)

    # From processed batches
    processed_dir = root / "data" / "processed"
    if processed_dir.exists():
        for batch_file in processed_dir.glob("batch_*.jsonl"):
            with batch_file.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        c_id = record.get("channel")
                        if c_id:
                            channel_ids.add(c_id)
                    except Exception:
                        pass
    return channel_ids


def fetch_channel_details(client: YouTubeClient, channel_ids: list[str]) -> dict[str, dict]:
    results = {}
    # Process in chunks of 50
    chunk_size = 50
    for i in range(0, len(channel_ids), chunk_size):
        chunk = channel_ids[i : i + chunk_size]
        ids_str = ",".join(chunk)
        request = client._youtube.channels().list(part="snippet", id=ids_str)
        response = client._execute(request, cost=1)
        for item in response.get("items", []):
            c_id = item["id"]
            snippet = item.get("snippet", {})
            results[c_id] = {
                "channel_id": c_id,
                "title": snippet.get("title", ""),
                "custom_url": snippet.get("customUrl", ""),
                "description": snippet.get("description", "")[:200],
                "published_at": snippet.get("publishedAt", ""),
            }
    return results


def main() -> None:
    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        raise SystemExit("YOUTUBE_API_KEY not set in .env")

    state = State(ROOT / "data" / "state" / "scrape_state.json")
    client = YouTubeClient(api_key, state)

    print("Collecting unique channel IDs...")
    channel_ids = sorted(list(collect_unique_channel_ids(ROOT, state)))
    print(f"Found {len(channel_ids)} unique channel IDs.")

    output_path = ROOT / "data" / "state" / "channel_names.json"
    existing_map = {}
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            existing_map = json.load(f)

    to_fetch = [cid for cid in channel_ids if cid not in existing_map]
    print(f"Fetching details for {len(to_fetch)} channel IDs via YouTube API...")

    if to_fetch:
        new_details = fetch_channel_details(client, to_fetch)
        existing_map.update(new_details)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(existing_map, f, ensure_ascii=False, indent=2)
        print(f"Updated {output_path} with {len(existing_map)} channels.")
    else:
        print(f"All {len(channel_ids)} channels are already in {output_path}.")

    print(f"Quota used today: {state.data['quota']['units_used']} / 10000")


if __name__ == "__main__":
    main()
