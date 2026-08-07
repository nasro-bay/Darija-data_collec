"""Recursively discovers djelfa.info's full subforum tree via BFS over
forumdisplay.php pages, starting from the forum index. Flags
permission-denied ("private") forums encountered along the way rather
than guessing from naming — see parse.py's module docstring for why.
"""
from __future__ import annotations

from typing import Optional

from bs4 import BeautifulSoup
from tqdm import tqdm

from .http_client import ForumHttpClient
from .parse import category_subforum_scope, extract_category_headers, extract_forum_links, looks_like_permission_denied

FORUM_INDEX_URL = "https://www.djelfa.info/vb/"
FORUM_DISPLAY_URL = "https://www.djelfa.info/vb/forumdisplay.php?f={forum_id}"


def discover_forum_tree(client: ForumHttpClient, *, max_forums: Optional[int] = None) -> dict[str, dict]:
    """Returns {forum_id: {title, url, parent_id, category_id,
    category_title, is_private, is_category}}, covering every subforum
    reachable from the index (arbitrary nesting depth), skipping (but
    still recording) any that turn out permission-denied.
    """
    index_resp = client.get(FORUM_INDEX_URL)
    index_soup = BeautifulSoup(index_resp.text, "lxml")

    forums: dict[str, dict] = {}
    queue: list[str] = []

    for category_id, category_title in extract_category_headers(index_soup):
        forums[category_id] = {
            "title": category_title,
            "url": FORUM_DISPLAY_URL.format(forum_id=category_id),
            "parent_id": None,
            "category_id": category_id,
            "category_title": category_title,
            "is_private": False,
            "is_category": True,
        }
        scope = category_subforum_scope(index_soup, category_id)
        if scope is None:
            continue
        for link in extract_forum_links(scope):
            if link.forum_id in forums:
                continue
            forums[link.forum_id] = {
                "title": link.title,
                "url": FORUM_DISPLAY_URL.format(forum_id=link.forum_id),
                "parent_id": category_id,
                "category_id": category_id,
                "category_title": category_title,
                "is_private": False,
                "is_category": False,
            }
            queue.append(link.forum_id)

    visited = set(forums.keys())
    progress = tqdm(desc="discovering subforums", unit="forum")
    try:
        while queue:
            if max_forums is not None and len(forums) >= max_forums:
                break
            forum_id = queue.pop(0)
            parent_info = forums[forum_id]

            resp = client.get(FORUM_DISPLAY_URL.format(forum_id=forum_id))
            if looks_like_permission_denied(resp.text):
                parent_info["is_private"] = True
                progress.update(1)
                continue

            soup = BeautifulSoup(resp.text, "lxml")
            for link in extract_forum_links(soup):
                if link.forum_id in visited:
                    continue
                visited.add(link.forum_id)
                forums[link.forum_id] = {
                    "title": link.title,
                    "url": FORUM_DISPLAY_URL.format(forum_id=link.forum_id),
                    "parent_id": forum_id,
                    "category_id": parent_info.get("category_id"),
                    "category_title": parent_info.get("category_title"),
                    "is_private": False,
                    "is_category": False,
                }
                queue.append(link.forum_id)
            progress.update(1)
    finally:
        progress.close()

    return forums
