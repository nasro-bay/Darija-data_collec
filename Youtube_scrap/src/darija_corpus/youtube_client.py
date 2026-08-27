"""Thin, quota-aware wrapper over the YouTube Data API v3.

Every request goes through `_execute`, which spends units against the
shared `State` (raising `QuotaExceededError` before making a call that
would blow the daily 10,000-unit budget), and retries transient errors
with exponential backoff.
"""
from __future__ import annotations

import json
import time
from typing import Optional

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from .state import QuotaExceededError, State

RETRYABLE_REASONS = {"rateLimitExceeded", "userRateLimitExceeded", "backendError", "internalError"}
MAX_RETRIES = 5

COST_VIDEOS_LIST = 1
COST_CHANNELS_LIST = 1
COST_PLAYLIST_ITEMS_LIST = 1
COST_COMMENT_THREADS_LIST = 1
COST_COMMENTS_LIST = 1
COST_SEARCH_LIST = 100


def error_reason(exc: HttpError) -> Optional[str]:
    try:
        content = exc.content
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")
        return json.loads(content)["error"]["errors"][0]["reason"]
    except Exception:
        return None


class YouTubeClient:
    def __init__(self, api_key: str, state: State):
        self._youtube = build("youtube", "v3", developerKey=api_key)
        self._state = state

    def _execute(self, request, cost: int) -> dict:
        self._state.spend(cost)
        last_error: Optional[BaseException] = None
        for attempt in range(MAX_RETRIES):
            try:
                return request.execute()
            except HttpError as exc:
                last_error = exc
                reason = error_reason(exc)
                status = exc.resp.status if exc.resp is not None else None
                if reason == "quotaExceeded":
                    raise QuotaExceededError(
                        "YouTube API reported the daily quota is exhausted"
                    ) from exc
                if reason in RETRYABLE_REASONS or (status is not None and status >= 500):
                    time.sleep(2**attempt)
                    continue
                raise
            except OSError as exc:
                # Network-level failure (connection timeout/reset/refused,
                # DNS hiccup, etc.) -- happens before any HTTP response is
                # received, so it never reaches the HttpError branch above.
                # Confirmed to crash the whole scrape run uncaught otherwise
                # (a raw TimeoutError propagating out of request.execute())
                # -- always worth retrying like the other transient cases
                # above.
                last_error = exc
                time.sleep(2**attempt)
                continue
        raise last_error

    def resolve_channel(
        self, *, channel_id: Optional[str] = None, handle: Optional[str] = None
    ) -> Optional[dict]:
        """Resolves a channel id/handle to its id, uploads-playlist id, title, and customUrl (1 unit).

        The uploads playlist is returned newest-video-first by
        `playlistItems.list`, which is what makes newest-first channel
        walks cheap (no need for `search.list(order=date)`).
        """
        if channel_id:
            request = self._youtube.channels().list(part="snippet,contentDetails", id=channel_id)
        elif handle:
            request = self._youtube.channels().list(part="snippet,contentDetails", forHandle=handle)
        else:
            return None
        response = self._execute(request, COST_CHANNELS_LIST)
        items = response.get("items", [])
        if not items:
            return None
        snippet = items[0].get("snippet", {})
        uploads_playlist_id = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]
        return {
            "channel_id": items[0]["id"],
            "uploads_playlist_id": uploads_playlist_id,
            "title": snippet.get("title", ""),
            "custom_url": snippet.get("customUrl", ""),
        }

    def list_playlist_video_ids(
        self, playlist_id: str, page_token: Optional[str] = None
    ) -> tuple[list[str], Optional[str]]:
        request = self._youtube.playlistItems().list(
            part="contentDetails",
            playlistId=playlist_id,
            maxResults=50,
            pageToken=page_token,
        )
        response = self._execute(request, COST_PLAYLIST_ITEMS_LIST)
        video_ids = [item["contentDetails"]["videoId"] for item in response.get("items", [])]
        return video_ids, response.get("nextPageToken")

    def get_video_metadata(self, video_id: str) -> Optional[dict]:
        request = self._youtube.videos().list(part="snippet", id=video_id)
        response = self._execute(request, COST_VIDEOS_LIST)
        items = response.get("items", [])
        if not items:
            return None
        snippet = items[0]["snippet"]
        return {
            "channel_id": snippet["channelId"],
            "channel_title": snippet.get("channelTitle", ""),
            "title": snippet.get("title", ""),
        }

    def list_comment_threads(self, video_id: str, page_token: Optional[str] = None) -> dict:
        request = self._youtube.commentThreads().list(
            part="snippet,replies",
            videoId=video_id,
            maxResults=100,
            pageToken=page_token,
            textFormat="plainText",
        )
        return self._execute(request, COST_COMMENT_THREADS_LIST)

    def list_replies(self, parent_id: str, page_token: Optional[str] = None) -> dict:
        request = self._youtube.comments().list(
            part="snippet",
            parentId=parent_id,
            maxResults=100,
            pageToken=page_token,
            textFormat="plainText",
        )
        return self._execute(request, COST_COMMENTS_LIST)

    def search_channels(
        self, query: str, *, region_code: str = "DZ", max_results: int = 25
    ) -> list[dict]:
        request = self._youtube.search().list(
            part="snippet",
            type="channel",
            q=query,
            regionCode=region_code,
            relevanceLanguage="ar",
            maxResults=max_results,
        )
        response = self._execute(request, COST_SEARCH_LIST)
        return response.get("items", [])
