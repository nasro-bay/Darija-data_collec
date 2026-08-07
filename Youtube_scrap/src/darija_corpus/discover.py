"""Discover candidate Algerian YouTube channels via search.list, for a
human to review and add to config/seed_channels.yaml. Never auto-adds
anything — this is a curation aid, not an auto-scraper.
"""
from __future__ import annotations

from typing import Optional

from .youtube_client import YouTubeClient

DEFAULT_KEYWORDS = [
    "فلوق جزائري",
    "دارجة جزائرية",
    "كوميدي جزائري",
    "comédie algérienne",
    "vlog algérien",
    "جزائر بودكاست",
]


def discover_channels(
    client: YouTubeClient,
    keywords: Optional[list[str]] = None,
    max_results_per_keyword: int = 10,
) -> list[dict]:
    keywords = keywords or DEFAULT_KEYWORDS
    seen_ids: set[str] = set()
    candidates: list[dict] = []
    for keyword in keywords:
        items = client.search_channels(keyword, max_results=max_results_per_keyword)
        for item in items:
            channel_id = item.get("id", {}).get("channelId")
            if not channel_id or channel_id in seen_ids:
                continue
            seen_ids.add(channel_id)
            snippet = item.get("snippet", {})
            candidates.append(
                {
                    "channel_id": channel_id,
                    "title": snippet.get("title", ""),
                    "description": snippet.get("description", ""),
                    "matched_keyword": keyword,
                }
            )
    return candidates
