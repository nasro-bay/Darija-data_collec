"""Scrape raw YouTube comments into data/raw/<video_id>.jsonl, either from
individually-specified video links (`scrape_video`, or in parallel via
`scrape_videos_parallel`) or by walking a whole channel's uploads
newest-first (`scrape_channel`) — all funnel into the same per-video
comment-fetch logic, so a video reached any way is never scraped twice.
Resumable at the channel, video, and comment-page level via the shared
`State`. Video info (video_id, channel, scrape_date) is carried only as
fields on each comment record — no separate video-level record is kept.
"""
from __future__ import annotations

import json
import os
import re
from datetime import date
from multiprocessing import Pool
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

from googleapiclient.errors import HttpError

from .state import QuotaExceededError, State
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


class _LocalQuotaCap:
    """Minimal State-compatible quota tracker (`.spend()` raising
    QuotaExceededError) scoped to one parallel worker's slice of the daily
    budget -- lets a worker's real YouTubeClient be quota-capped through
    the exact same mechanism the sequential path uses (State.spend()),
    without touching any file or the real persisted daily total.
    """

    def __init__(self, unit_cap: int):
        self.units_used = 0
        self._cap = unit_cap

    def spend(self, units: int) -> None:
        if self.units_used + units > self._cap:
            raise QuotaExceededError(f"worker unit cap reached ({self.units_used}+{units} > {self._cap})")
        self.units_used += units


def _scrape_one_video_core(
    client, video_id: str, channel_id: Optional[str], resume_page_token: Optional[str]
) -> dict:
    """Scrapes one video's comments using `client` (real or fake, anything
    satisfying YouTubeClient's interface) -- deliberately independent of
    any shared State, which is what makes it safe to run inside a
    multiprocessing worker (see _scrape_one_video_worker) *and* directly
    in a test with a FakeClient, no multiprocessing involved.

    Returns a self-contained result for the caller to apply to State:
      {"video_id", "channel_id", "channel_title", "status", "records",
       "next_page_token", "reason"?}
    status: "done" | "partial" (stopped early -- quota cap hit, resume via
    next_page_token) | "comments_disabled" | "error" (reason set).

    Any HttpError (including an unexpected/non-skippable one) is caught
    here rather than re-raised, unlike the sequential scrape_video_comments
    -- one bad video must not crash sibling workers' already-completed
    results in the same parallel batch.
    """
    channel_title = None
    page_token = resume_page_token
    records: list[dict] = []
    try:
        if channel_id is None:
            metadata = client.get_video_metadata(video_id)
            if not metadata:
                return {
                    "video_id": video_id, "channel_id": None, "channel_title": None,
                    "status": "error", "records": [], "next_page_token": None,
                    "reason": "video not found",
                }
            channel_id = metadata["channel_id"]
            channel_title = metadata.get("channel_title")

        while True:
            response = client.list_comment_threads(video_id, page_token=page_token)
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
            page_token = response.get("nextPageToken")
            if not page_token:
                break
        return {
            "video_id": video_id, "channel_id": channel_id, "channel_title": channel_title,
            "status": "done", "records": records, "next_page_token": None,
        }
    except QuotaExceededError:
        return {
            "video_id": video_id, "channel_id": channel_id, "channel_title": channel_title,
            "status": "partial", "records": records, "next_page_token": page_token,
        }
    except HttpError as exc:
        reason = error_reason(exc)
        if reason in SKIPPABLE_REASONS:
            return {
                "video_id": video_id, "channel_id": channel_id, "channel_title": channel_title,
                "status": "comments_disabled", "records": records, "next_page_token": None,
            }
        return {
            "video_id": video_id, "channel_id": channel_id, "channel_title": channel_title,
            "status": "error", "records": records, "next_page_token": page_token, "reason": str(exc),
        }


def _scrape_one_video_worker(args: dict) -> dict:
    """multiprocessing.Pool entry point -- must be module-level so it's
    picklable, including Windows' spawn start method (same requirement as
    pipeline.py's worker functions). Builds a real, per-worker-capped
    YouTubeClient and delegates to _scrape_one_video_core; never touches
    the shared State or writes any file itself -- see
    scrape_videos_parallel's docstring for why.
    """
    quota = _LocalQuotaCap(args["unit_cap"])
    client = YouTubeClient(args["api_key"], quota)
    result = _scrape_one_video_core(client, args["video_id"], args["channel_id"], args["resume_page_token"])
    result["units_spent"] = quota.units_used
    return result


def scrape_videos_parallel(
    api_key: str,
    state: State,
    raw_dir: Path,
    video_url_or_ids: list[str],
    *,
    workers: Optional[int] = None,
    worker_fn=_scrape_one_video_worker,
    map_fn=None,
) -> list[dict]:
    """Scrapes a list of independent video links/IDs in parallel across a
    process pool -- each worker handles a DIFFERENT video's full comment
    fetch concurrently. This is what run_scrape.py's seed_videos.yaml loop
    uses.

    All State mutation and raw-file writes happen back in this (main)
    process only, once per batch, after every worker in that batch has
    returned -- workers never touch the shared state.json or write raw
    files themselves. This is deliberate: state.json is a single shared
    file (see state.py's docstring), and multiple processes each loading
    their own in-memory copy and calling .save() independently would race
    and silently lose each other's updates (last save wins). Keeping every
    write sequential in the main process avoids that entirely, the same
    pattern pipeline.py already uses for its own process pool.

    Quota safety: real per-request spending happens inside each worker's
    own YouTubeClient, capped locally to that worker's fair share of the
    remaining daily budget for the batch (_LocalQuotaCap) -- so a batch of
    workers can never collectively spend more than what was remaining when
    the batch was dispatched, even though the real state.json ledger is
    only updated afterward, in the main process. (One edge case: very
    close to quota exhaustion, integer-dividing a tiny remaining budget
    across a full batch can round each worker's cap up to a minimum of 1,
    letting a batch overshoot the true remaining budget by up to
    len(batch)-1 units in the worst case -- negligible against the
    10,000-unit daily budget, and Google's own server-side enforcement,
    already handled via QuotaExceededError, is the real backstop.)

    Trade-off vs. the sequential scrape_video(): a video dispatched to a
    worker is no longer resumable at the comment-page level if the whole
    process is killed mid-video -- only at the batch boundary. A worker
    hitting its own quota cap still reports "partial" + a resume cursor
    (same as scrape_video hitting the real daily quota), so a later run
    picks it back up from that page, not from scratch; only a hard kill
    mid-worker loses that one video's progress. Acceptable given seed
    videos are typically far smaller than a full channel walk.

    `worker_fn`/`map_fn` are injectable for tests (bypass real
    multiprocessing and a real API key) -- production code should never
    pass them.
    """
    num_workers = workers if workers is not None else (os.cpu_count() or 1)

    pending: list[tuple[str, Optional[str]]] = []
    results: list[dict] = []
    for entry in video_url_or_ids:
        video_id = extract_video_id(entry)
        video_state = state.video_state(video_id)
        if video_state["status"] in ("done", "comments_disabled"):
            results.append(
                {"video_id": video_id, "channel": video_state.get("channel_id"), "status": video_state["status"]}
            )
            continue
        pending.append((video_id, video_state.get("channel_id")))

    for batch_start in range(0, len(pending), num_workers):
        batch = pending[batch_start : batch_start + num_workers]
        remaining = state.units_remaining()
        if remaining <= 0:
            for video_id, channel_id in batch:
                results.append({"video_id": video_id, "channel": channel_id, "status": "quota_exceeded"})
            break

        per_worker_cap = max(1, remaining // len(batch))
        args_list = [
            {
                "api_key": api_key,
                "video_id": video_id,
                "channel_id": channel_id,
                "resume_page_token": state.video_state(video_id).get("next_comment_page_token"),
                "unit_cap": per_worker_cap,
            }
            for video_id, channel_id in batch
        ]

        if map_fn is not None:
            batch_results = list(map_fn(worker_fn, args_list))
        else:
            with Pool(processes=min(num_workers, len(batch))) as pool:
                batch_results = pool.map(worker_fn, args_list)

        for r in batch_results:
            video_state = state.video_state(r["video_id"])
            if r.get("channel_id"):
                video_state["channel_id"] = r["channel_id"]
                if r.get("channel_title"):
                    state.record_channel_info(r["channel_id"], r["channel_title"])
            _append_jsonl(raw_dir / f"{r['video_id']}.jsonl", r["records"])
            state.spend(r["units_spent"])

            status = r["status"]
            if status in ("done", "comments_disabled"):
                video_state["status"] = status
                video_state["next_comment_page_token"] = None
            elif status == "partial":
                video_state["status"] = "pending"
                video_state["next_comment_page_token"] = r["next_page_token"]
            else:  # "error"
                video_state["status"] = "error"

            entry = {"video_id": r["video_id"], "channel": r.get("channel_id"), "status": status}
            if "reason" in r:
                entry["reason"] = r["reason"]
            results.append(entry)

        state.save()

    return results


def scrape_channel(
    client: YouTubeClient,
    state: State,
    raw_dir: Path,
    *,
    channel_id: Optional[str] = None,
    handle: Optional[str] = None,
    max_videos: Optional[int] = None,
    api_key: Optional[str] = None,
    workers: Optional[int] = None,
    worker_fn=None,
    map_fn=None,
) -> dict:
    """Walks a channel's uploads playlist newest-video-first. If `max_videos`
    is set, stops once that many of the newest videos have been considered
    (persisted across runs, so a resumed walk won't overshoot the cap).
    Which videos to process each page is always decided sequentially, cheaply,
    with no API calls (just cap bookkeeping against the already-fetched page
    of video IDs) -- newest-first selection order is a correctness
    requirement (it's what makes `max_videos` mean "the N newest", and what
    `capped_at`/resume bookkeeping assumes), but the actual, expensive
    comment-fetching for that page's selected videos does not need to
    happen in that same order to still be correct.

    If `api_key` is given, each page's selected videos are scraped in
    PARALLEL across a process pool (via scrape_videos_parallel) -- pass
    `workers` to control pool size. If `api_key` is omitted (the default),
    falls back to the original sequential behavior via
    `scrape_video_comments` using `client` directly, unchanged -- this is
    what every existing caller/test still gets.

    Behavioral difference in the parallel path: quota exhaustion mid-walk
    is reported back in the returned dict (`"quota_exceeded": True`)
    instead of raising QuotaExceededError -- multiprocessing.Pool can't
    propagate a single worker's exception without discarding its
    already-collected sibling results, so scrape_videos_parallel always
    returns rather than raises (see its docstring); scrape_channel's
    sequential path keeps raising, unchanged, since existing callers/tests
    depend on that.
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
    quota_stopped = False
    while True:
        video_ids, next_token = client.list_playlist_video_ids(playlist_id, page_token=page_token)

        page_selected: list[str] = []
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
            page_selected.append(video_id)

        if api_key is not None:
            # Known channel_id -- pre-seed it so scrape_videos_parallel's
            # per-worker resolution skips a redundant metadata lookup.
            for video_id in page_selected:
                state.video_state(video_id)["channel_id"] = resolved_id
            page_results = scrape_videos_parallel(
                api_key, state, raw_dir, page_selected,
                workers=workers, worker_fn=worker_fn or _scrape_one_video_worker, map_fn=map_fn,
            )
            for r in page_results:
                if r["status"] == "done":
                    videos_done += 1
                elif r["status"] == "comments_disabled":
                    videos_skipped += 1
                elif r["status"] == "quota_exceeded":
                    quota_stopped = True
        else:
            for video_id in page_selected:
                status = scrape_video_comments(client, state, raw_dir, video_id, resolved_id)
                if status == "done":
                    videos_done += 1
                elif status == "comments_disabled":
                    videos_skipped += 1

        if not hit_cap and not quota_stopped:
            # Only advance the resume cursor past this page if we finished
            # it — if capped/quota-stopped mid-page, leave it pointing at
            # this same page, so a later run re-fetches it and continues
            # from where it stopped instead of skipping videos.
            channel_state["next_playlist_page_token"] = next_token
        state.save()
        if hit_cap or quota_stopped:
            break
        page_token = next_token
        if not page_token:
            break

    channel_state["capped_at"] = max_videos if hit_cap else None
    channel_state["completed"] = not hit_cap and not quota_stopped
    state.save()
    result = {
        "channel": resolved_id,
        "videos_considered": channel_state["videos_found"],
        "videos_done": videos_done,
        "videos_skipped": videos_skipped,
    }
    if quota_stopped:
        result["quota_exceeded"] = True
    return result
