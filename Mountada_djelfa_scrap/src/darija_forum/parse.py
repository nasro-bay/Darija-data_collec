"""Parses djelfa.info's vBulletin forum-index / subforum pages to
discover the subforum tree. Selectors verified against a real saved page
(fetched via a manually-bootstrapped session), not guessed blindly from
the generic vBulletin pattern:

- Genuine category headers are `td.tcat` containing a real
  `forumdisplay.php?f=ID` link — vBulletin reuses `td.tcat` for other
  widgets too ("آخر المواضيع", bare "#top" collapse toggles), which this
  filters out.
- A category's top-level subforums live in
  `tbody#collapseobj_forumbit_<category_id>`, each subforum in its own
  `td[id="f<ID>"]`.
- Deeper-nested child forums appear as inline links inside a parent's
  "الأقسام الفرعية" (sub-sections) block, not as their own top-level
  `td[id]` row — full recursion (visiting each forum's own page) is
  needed to reach them, see `discover.py`.
- The word "خاص" ("dedicated to") shows up constantly in ordinary
  subforum names/descriptions and is NOT a private/restricted-access
  marker on this site — access restriction is detected at *visit time*
  (a permission-denied response), not from naming.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from bs4 import BeautifulSoup

FORUM_ID_RE = re.compile(r"forumdisplay\.php\?f=(\d+)")

# Best-effort permission-denied markers. Not verified against a real
# restricted forum yet (none encountered during initial discovery) —
# expand this if discovery run mislabels/misses a truly restricted
# section; treat unexpectedly-empty forum pages with suspicion too.
PERMISSION_DENIED_MARKERS = (
    "ليس لديك الصلاحية",
    "you do not have permission",
    "غير مصرح لك",
    "access denied",
)


THREAD_ID_RE = re.compile(r"showthread\.php\?t=(\d+)")
POST_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2},\s*\d{2}:\d{2}")
BASE_URL = "https://www.djelfa.info/vb/"


@dataclass
class ForumLink:
    forum_id: str
    title: str


@dataclass
class ThreadLink:
    thread_id: str
    title: str


@dataclass
class PostRecord:
    post_id: str
    author: Optional[str]
    timestamp: Optional[str]
    text: str
    post_url: str


def extract_category_headers(soup: BeautifulSoup) -> list[tuple[str, str]]:
    """Returns [(forum_id, title), ...] for genuine top-level categories."""
    headers = []
    for td in soup.select("td.tcat"):
        a = td.find("a", href=FORUM_ID_RE)
        if not a:
            continue
        match = FORUM_ID_RE.search(a["href"])
        title = a.get_text(strip=True)
        if match and title:
            headers.append((match.group(1), title))
    return headers


def extract_forum_links(soup: BeautifulSoup) -> list[ForumLink]:
    """Returns every forumdisplay.php?f=ID link found in `soup` (dedup'd
    by forum_id) — covers both top-level td[id=fNN] rows and inline
    "الأقسام الفرعية" child links. `soup` can be a whole page or a
    narrower scope (e.g. one category's tbody).
    """
    seen: dict[str, ForumLink] = {}
    for a in soup.find_all("a", href=FORUM_ID_RE):
        match = FORUM_ID_RE.search(a["href"])
        forum_id = match.group(1)
        title = a.get_text(strip=True)
        if not title or forum_id in seen:
            continue
        seen[forum_id] = ForumLink(forum_id=forum_id, title=title)
    return list(seen.values())


def category_subforum_scope(soup: BeautifulSoup, category_id: str) -> Optional[BeautifulSoup]:
    """Returns the tbody scoping a category's top-level subforums, or
    None if that category has no visible subforums (e.g. all private).
    """
    return soup.select_one(f"tbody#collapseobj_forumbit_{category_id}")


def looks_like_permission_denied(html: str) -> bool:
    lowered = html.lower()
    return any(marker.lower() in lowered for marker in PERMISSION_DENIED_MARKERS)


def _has_next_page(soup: BeautifulSoup) -> bool:
    """vBulletin's pagenav uses a reliable rel="next" link on the
    "next page" control when one exists — confirmed on real subforum and
    thread pages ("صفحة X من Y" / rel=next), simpler and more robust than
    counting/parsing numbered page links.
    """
    pagenav = soup.select_one("div.pagenav")
    if not pagenav:
        return False
    return pagenav.select_one('a[rel="next"]') is not None


def list_threads(html: str) -> tuple[list[ThreadLink], bool]:
    """Parses a subforum listing page (forumdisplay.php?f=ID&page=N) into
    its thread entries + whether a further page exists. Thread rows are
    identified via `a[id^="thread_title_"]` — a stable, unique-per-thread
    selector that naturally excludes per-thread mini pagination links,
    "last post" permalinks, and the site-wide registration announcement
    (none of which carry that id).
    """
    soup = BeautifulSoup(html, "lxml")
    threads: list[ThreadLink] = []
    seen: set[str] = set()
    for a in soup.select('a[id^="thread_title_"]'):
        thread_id = a["id"][len("thread_title_") :]
        title = a.get_text(strip=True)
        if not title or thread_id in seen:
            continue
        seen.add(thread_id)
        threads.append(ThreadLink(thread_id=thread_id, title=title))
    return threads, _has_next_page(soup)


def list_posts(html: str) -> tuple[list[PostRecord], bool]:
    """Parses a thread page (showthread.php?t=ID&page=N) into its posts +
    whether a further page exists. Each post is a `table[id="post<ID>"]`
    containing `div#post_message_<ID>` (body text), `a.bigusername`
    (author — may be absent for a deleted/guest account), and a
    `td.thead` with the post date ("YYYY-MM-DD, HH:MM").
    """
    soup = BeautifulSoup(html, "lxml")
    posts: list[PostRecord] = []
    for table in soup.select('table[id^="post"]'):
        post_id = table.get("id", "")[len("post") :]
        if not post_id.isdigit():
            continue  # e.g. a nested "table52"-style id, not a post container
        message_div = table.select_one(f"div#post_message_{post_id}")
        if message_div is None:
            continue
        text = message_div.get_text("\n", strip=True)
        if not text:
            continue
        author_el = table.select_one("a.bigusername")
        thead = table.select_one("td.thead")
        date_match = POST_DATE_RE.search(thead.get_text()) if thead else None
        posts.append(
            PostRecord(
                post_id=post_id,
                author=author_el.get_text(strip=True) if author_el else None,
                timestamp=date_match.group(0) if date_match else None,
                text=text,
                post_url=f"{BASE_URL}showthread.php?p={post_id}#post{post_id}",
            )
        )
    return posts, _has_next_page(soup)
