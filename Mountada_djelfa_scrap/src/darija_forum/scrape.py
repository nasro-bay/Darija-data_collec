"""Scrapes djelfa.info forum threads/posts into
data/raw/djelfa/<subforum_id>/<thread_id>.jsonl. Resumable at the
subforum, thread, and post-page level via the shared `State`. Uses the
same cap-now-resume-later semantics (`capped_at`/`completed`) proven for
Youtube_scrap's channel walker: stopping at `max_threads` doesn't block
resuming further later with a higher (or no) cap.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Optional

from .http_client import ForumHttpClient
from .parse import list_posts, list_threads
from .state import State

BASE_URL = "https://www.djelfa.info/vb/"

# Safety net for scrape_thread's new-posts guard (see below): a real
# forum thread this long (7500+ posts) is essentially unheard of.
MAX_POST_PAGES = 500


def _append_jsonl(path: Path, records: list[dict]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _existing_post_ids(raw_path: Path) -> set:
    if not raw_path.exists():
        return set()
    ids = set()
    with raw_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["post_id"])
    return ids


def _post_dict(*, post, thread_id: str, subforum_id: str, thread_title: str) -> dict:
    return {
        "post_id": post.post_id,
        "text": post.text,
        "thread_id": thread_id,
        "thread_title": thread_title,
        "thread_url": f"{BASE_URL}showthread.php?t={thread_id}",
        "subforum_id": subforum_id,
        "author": post.author,
        "timestamp": post.timestamp,
        "post_url": post.post_url,
        "scrape_date": date.today().isoformat(),
    }


def scrape_thread(
    client: ForumHttpClient, state: State, raw_dir: Path, thread_id: str, subforum_id: str, thread_title: str
) -> str:
    """Pages through one thread's posts. Returns 'done' (SessionExpiredError propagates).

    Stops as soon as a page yields no *new* posts, even if the site's own
    pagination signals ("there's a next page") say otherwise. Confirmed
    on real data: some single-page threads keep reporting a next page
    forever (a stray `rel="next"` on the page unrelated to real post
    pagination, or the site clamping out-of-range page requests back to
    page 1) — without this guard, that re-fetches and re-appends the same
    handful of posts indefinitely. `MAX_POST_PAGES` is a hard backstop in
    case some other failure mode keeps yielding "new" content forever.
    """
    thread_state = state.thread_state(thread_id)
    thread_state["subforum_id"] = subforum_id
    thread_state["title"] = thread_title
    if thread_state["status"] == "done":
        return "done"

    raw_path = raw_dir / subforum_id / f"{thread_id}.jsonl"
    seen_post_ids = _existing_post_ids(raw_path)
    page = thread_state.get("next_post_page", 1)

    while True:
        resp = client.get(f"{BASE_URL}showthread.php?t={thread_id}&page={page}")
        posts, has_next = list_posts(resp.text)
        new_posts = [p for p in posts if p.post_id not in seen_post_ids]

        if not new_posts:
            break

        seen_post_ids.update(p.post_id for p in new_posts)
        records = [
            _post_dict(post=p, thread_id=thread_id, subforum_id=subforum_id, thread_title=thread_title)
            for p in new_posts
        ]
        _append_jsonl(raw_path, records)

        next_page = page + 1 if has_next else page
        thread_state["next_post_page"] = next_page
        state.save()

        if not has_next:
            break
        if page >= MAX_POST_PAGES:
            print(
                f"WARNING: thread {thread_id} hit the {MAX_POST_PAGES}-page safety cap "
                "while still finding new posts each page — stopping early. Worth checking "
                "whether this thread is legitimately huge."
            )
            break
        page = next_page

    thread_state["status"] = "done"
    state.save()
    return "done"


def scrape_subforum(
    client: ForumHttpClient,
    state: State,
    raw_dir: Path,
    forum_id: str,
    *,
    max_threads: Optional[int] = None,
) -> dict:
    """Pages through a subforum's thread listing (server default order —
    most-recently-active first), scraping each thread's posts.
    `SessionExpiredError` propagates up (state is saved incrementally, so
    a rerun after refreshing the session resumes cleanly).
    """
    subforum_state = state.subforum_state(forum_id)

    if subforum_state["completed"]:
        return {
            "forum_id": forum_id,
            "threads_considered": subforum_state["threads_found"],
            "note": "already completed",
        }

    prior_cap = subforum_state.get("capped_at")
    if prior_cap is not None and max_threads is not None and max_threads <= prior_cap:
        return {
            "forum_id": forum_id,
            "threads_considered": subforum_state["threads_found"],
            "note": f"already scraped up to its cap ({prior_cap} threads) — raise max_threads to continue",
        }

    counted_ids = set(subforum_state["counted_thread_ids"])
    page = subforum_state.get("next_thread_page", 1)
    threads_done = 0
    hit_cap = False

    while True:
        resp = client.get(f"{BASE_URL}forumdisplay.php?f={forum_id}&page={page}")
        threads, has_next = list_threads(resp.text)
        for thread in threads:
            if max_threads is not None and subforum_state["threads_found"] >= max_threads:
                hit_cap = True
                break
            if thread.thread_id not in counted_ids:
                # Guards against double-counting a page re-fetched on
                # resume after a crash mid-page (same reasoning as the
                # YouTube channel walker's counted_video_ids).
                counted_ids.add(thread.thread_id)
                subforum_state["counted_thread_ids"].append(thread.thread_id)
                subforum_state["threads_found"] += 1
            status = scrape_thread(client, state, raw_dir, thread.thread_id, forum_id, thread.title)
            if status == "done":
                threads_done += 1

        next_page = page + 1 if has_next else page
        if not hit_cap:
            # Only advance the resume cursor past this page if we
            # finished it — if capped mid-page, leave it pointing at this
            # same page so a later, higher-cap run re-fetches it and
            # continues from where it stopped instead of skipping threads.
            subforum_state["next_thread_page"] = next_page
        state.save()
        if hit_cap or not has_next:
            break
        page = next_page

    subforum_state["capped_at"] = max_threads if hit_cap else None
    subforum_state["completed"] = not hit_cap
    state.save()
    return {
        "forum_id": forum_id,
        "threads_considered": subforum_state["threads_found"],
        "threads_done": threads_done,
    }
