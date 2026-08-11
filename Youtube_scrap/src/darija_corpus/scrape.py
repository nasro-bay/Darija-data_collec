"""Scrape raw YouTube comments into data/raw/<video_id>.jsonl, either from
individually-specified video links (`scrape_video`) or by walking a whole
channel's uploads newest-first (`scrape_channel`) — both funnel into the
same per-video `scrape_video_comments`, so a video reached either way is
never scraped twice. Resumable at the channel, video, and comment-page
level via the shared `State`. Video info (video_id, channel, scrape_date)
is carried only as fields on each comment record — no separate
video-level record is kept.
"""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

from googleapiclient.errors import HttpError

from .state import State
from .youtube_client import YouTubeClient, error_reason

SKIPPABLE_REASONS = {"commentsDisabled", "videoNotFound"}
_CHANNEL_ID_RE = re.compile(r"^UC[\w-]{22}$")


def extract_video_id(url_or_id: str) -> str:
    """Accepts a bare video ID or any common YouTube URL shape
    (watch?v=, youtu.be/, /shorts/, /embed/, /live/) and returns the
    11-character video ID.
    """
    value = url_or_id.strip()
    if "youtube.com" not in value and "youtu.be" not in value:
        return value  # already a bare video ID

    parsed = urlparse(value)
    if parsed.hostname in ("youtu.be", "www.youtu.be"):
        return parsed.path.lstrip("/").split("/")[0]

    if parsed.path == "/watch":
        query_id = parse_qs(parsed.query).get("v")
        if query_id:
            return query_id[0]

    for prefix in ("/shorts/", "/embed/", "/live/"):
        if parsed.path.startswith(prefix):
            return parsed.path[len(prefix) :].split("/")[0]

    return parsed.path.rstrip("/").split("/")[-1]


def extract_channel_ref(value: str) -> dict:
    """Accepts a channel URL (/@handle, /channel/UCxxx, /c/Name, /user/Name),
    a bare @handle, or a bare channel ID, and returns `{"channel_id": ...}`
    or `{"handle": ...}` — whichever `resolve_channel()` needs.
    """
    value = value.strip()

    if "youtube.com" in value:
        segments = [s for s in urlparse(value).path.split("/") if s]
        if segments:
            if segments[0] == "channel" and len(segments) > 1:
                return {"channel_id": segments[1]}
            if segments[0].startswith("@"):
                return {"handle": segments[0]}
            if segments[0] in ("c", "user") and len(segments) > 1:
                return {"handle": f"@{segments[1]}"}
        return {}

    if _CHANNEL_ID_RE.match(value):
        return {"channel_id": value}

    return {"handle": value if value.startswith("@") else f"@{value}"}


def _append_jsonl(path: Path, records: list[dict]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _comment_record(
    *, comment_id: str, text: str, video_id: str, channel_id: str, source_type: str, parent_id: Optional[str] = None
) -> dict:
    return {
        "comment_id": comment_id,
        "text": text,
        "video_id": video_id,
        "channel": channel_id,
        "scrape_date": date.today().isoformat(),
        "source_type": source_type,
        "parent_id": parent_id,
    }


def _fetch_remaining_replies(
    client: YouTubeClient, parent_id: str, video_id: str, channel_id: str, skip: int = 0
) -> list[dict]:
    records: list[dict] = []
    page_token = None
    seen = 0
    while True:
        response = client.list_replies(parent_id, page_token=page_token)
        for reply in response.get("items", []):
            seen += 1
            if seen <= skip:
                continue
            records.append(
                _comment_record(
                    comment_id=reply["id"],
                    text=reply["snippet"]["textDisplay"],
                    video_id=video_id,
                    channel_id=channel_id,
                    source_type="reply",
                    parent_id=parent_id,
                )
            )
        page_token = response.get("nextPageToken")
        if not page_token:
            break
    return records


def scrape_video_comments(
    client: YouTubeClient, state: State, raw_dir: Path, video_id: str, channel_id: str
) -> str:
    """Scrapes all comments for one video. Returns 'done', 'comments_disabled', or 'error'."""
    video_state = state.video_state(video_id)
    video_state["channel_id"] = channel_id
    if video_state["status"] in ("done", "comments_disabled"):
        return video_state["status"]

    raw_path = raw_dir / f"{video_id}.jsonl"
    page_token = video_state.get("next_comment_page_token")
    try:
        while True:
            response = client.list_comment_threads(video_id, page_token=page_token)
            records: list[dict] = []
            for item in response.get("items", []):
                top = item["snippet"]["topLevelComment"]
                records.append(
                    _comment_record(
                        comment_id=top["id"],
                        text=top["snippet"]["textDisplay"],
                        video_id=video_id,
                        channel_id=channel_id,
                        source_type="comment",
                    )
                )
                total_replies = item["snippet"].get("totalReplyCount", 0)
                inline_replies = item.get("replies", {}).get("comments", [])
                for reply in inline_replies:
                    records.append(
                        _comment_record(
                            comment_id=reply["id"],
                            text=reply["snippet"]["textDisplay"],
                            video_id=video_id,
                            channel_id=channel_id,
                            source_type="reply",
                            parent_id=top["id"],
                        )
                    )
                if total_replies > len(inline_replies):
                    records.extend(
                        _fetch_remaining_replies(
                            client, top["id"], video_id, channel_id, skip=len(inline_replies)
                        )
                    )
            _append_jsonl(raw_path, records)
            page_token = response.get("nextPageToken")
            video_state["next_comment_page_token"] = page_token
            state.save()
            if not page_token:
                break
        video_state["status"] = "done"
        state.save()
        return "done"
    except HttpError as exc:
        reason = error_reason(exc)
        if reason in SKIPPABLE_REASONS:
            video_state["status"] = "comments_disabled"
            state.save()
            return "comments_disabled"
        video_state["status"] = "error"
        state.save()
        raise


def scrape_video(client: YouTubeClient, state: State, raw_dir: Path, video_url_or_id: str) -> dict:
    video_id = extract_video_id(video_url_or_id)
    video_state = state.video_state(video_id)

    channel_id = video_state.get("channel_id")
    if not channel_id:
        metadata = client.get_video_metadata(video_id)
        if not metadata:
            video_state["status"] = "error"
            state.save()
            return {"video_id": video_id, "status": "error", "reason": "video not found"}
        channel_id = metadata["channel_id"]
        video_state["channel_id"] = channel_id
        if metadata.get("channel_title"):
            state.record_channel_info(channel_id, metadata["channel_title"])
        state.save()

    status = scrape_video_comments(client, state, raw_dir, video_id, channel_id)
    return {"video_id": video_id, "channel": channel_id, "status": status}


def scrape_channel(
    client: YouTubeClient,
    state: State,
    raw_dir: Path,
    *,
    channel_id: Optional[str] = None,
    handle: Optional[str] = None,
    max_videos: Optional[int] = None,
) -> dict:
    """Walks a channel's uploads playlist newest-video-first, scraping each
    video's comments via `scrape_video_comments`. If `max_videos` is set,
    stops once that many of the newest videos have been considered
    (persisted across runs, so a resumed walk won't overshoot the cap).
    """
    # If we already know this channel's uploads playlist from a previous
    # run — whether it was passed as channel_id directly, or a handle we've
    # resolved before — skip re-resolving it (saves an API call every rerun).
    known_id = channel_id or (state.resolved_channel_id_for_handle(handle) if handle else None)
    cached = state.data["channels"].get(known_id) if known_id else None
    if cached and cached.get("uploads_playlist_id"):
        resolved_id = known_id
        channel_state = cached
    else:
        resolved = client.resolve_channel(channel_id=channel_id, handle=handle)
        if not resolved:
            return {"channel": channel_id or handle, "error": "channel not found"}
        resolved_id = resolved["channel_id"]
        if handle:
            state.remember_handle(handle, resolved_id)
        state.record_channel_info(resolved_id, resolved.get("title", ""), resolved.get("custom_url", ""))
        channel_state = state.channel_state(resolved_id)
        if not channel_state["uploads_playlist_id"]:
            channel_state["uploads_playlist_id"] = resolved["uploads_playlist_id"]
        state.save()  # persist the handle mapping even if the channel turns out already-completed
    playlist_id = channel_state["uploads_playlist_id"]

    videos_done = 0
    videos_skipped = 0

    if channel_state["completed"]:
        return {
            "channel": resolved_id,
            "videos_considered": channel_state["videos_found"],
            "videos_done": videos_done,
            "videos_skipped": videos_skipped,
            "note": "already completed",
        }

    prior_cap = channel_state.get("capped_at")
    if prior_cap is not None and max_videos is not None and max_videos <= prior_cap:
        return {
            "channel": resolved_id,
            "videos_considered": channel_state["videos_found"],
            "videos_done": videos_done,
            "videos_skipped": videos_skipped,
            "note": f"already scraped up to its cap ({prior_cap} videos) — raise max_videos to continue",
        }

    counted_ids = set(channel_state["counted_video_ids"])
    page_token = channel_state.get("next_playlist_page_token")
    hit_cap = False
    while True:
        video_ids, next_token = client.list_playlist_video_ids(playlist_id, page_token=page_token)
        for video_id in video_ids:
            if max_videos is not None and channel_state["videos_found"] >= max_videos:
                hit_cap = True
                break
            if video_id not in counted_ids:
                # Guards against double-counting: if a crash interrupted a
                # previous run partway through this same page, the page is
                # re-fetched from its start on resume, and videos already
                # counted (and possibly already scraped) would otherwise be
                # counted a second time against `max_videos`.
                counted_ids.add(video_id)
                channel_state["counted_video_ids"].append(video_id)
                channel_state["videos_found"] += 1
            status = scrape_video_comments(client, state, raw_dir, video_id, resolved_id)
            if status == "done":
                videos_done += 1
            elif status == "comments_disabled":
                videos_skipped += 1
        if not hit_cap:
            # Only advance the resume cursor past this page if we finished
            # it — if capped mid-page, leave it pointing at this same page,
            # so a later run with a higher (or no) cap re-fetches it and
            # continues from where it stopped instead of skipping videos.
            channel_state["next_playlist_page_token"] = next_token
        state.save()
        if hit_cap:
            break
        page_token = next_token
        if not page_token:
            break

    channel_state["capped_at"] = max_videos if hit_cap else None
    channel_state["completed"] = not hit_cap
    state.save()
    return {
        "channel": resolved_id,
        "videos_considered": channel_state["videos_found"],
        "videos_done": videos_done,
        "videos_skipped": videos_skipped,
    }
