"""Unit tests for scrape_subforum()/scrape_thread() using a fake HTTP
client — no live requests. Covers: pagination across thread-list and
post pages, the max_threads cap, resuming after a crash mid-page, a
completed subforum being a no-op on rerun, and raising the cap to
resume — same properties already proven for Youtube_scrap's channel
walker, ported to the forum-scraping shape.
"""
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from darija_forum.http_client import SessionExpiredError  # noqa: E402
from darija_forum.scrape import MAX_POST_PAGES, scrape_subforum, scrape_thread  # noqa: E402
from darija_forum.state import State  # noqa: E402


def _thread_list_html(entries: list[tuple], has_next: bool) -> str:
    links = "".join(
        f'<a id="thread_title_{tid}" href="showthread.php?t={tid}"><b>{title}</b></a>' for tid, title in entries
    )
    nav = '<div class="pagenav"><a rel="next" href="#">next</a></div>' if has_next else ""
    return f"<html><body>{links}{nav}</body></html>"


def _post_page_html(entries: list[tuple], has_next: bool) -> str:
    posts = "".join(
        f'<table id="post{pid}"><tr><td class="thead">2020-01-01, 00:00</td></tr>'
        f'<tr><td><a class="bigusername">user{pid}</a>'
        f'<div id="post_message_{pid}">{text}</div></td></tr></table>'
        for pid, text in entries
    )
    nav = '<div class="pagenav"><a rel="next" href="#">next</a></div>' if has_next else ""
    return f"<html><body>{posts}{nav}</body></html>"


class FakeResponse:
    def __init__(self, text: str):
        self.text = text
        self.status_code = 200


class FakeForumClient:
    """thread_pages: {forum_id: [page1_entries, page2_entries, ...]}
    post_pages: {thread_id: [page1_entries, page2_entries, ...]}
    Each entries list is [(id, title_or_text), ...].
    """

    def __init__(self, thread_pages: dict, post_pages: dict):
        self.thread_pages = thread_pages
        self.post_pages = post_pages
        self.calls: list[str] = []
        self.fail_after_calls: int = None

    def get(self, url: str) -> FakeResponse:
        self.calls.append(url)
        if self.fail_after_calls is not None and len(self.calls) > self.fail_after_calls:
            raise SessionExpiredError("simulated session expiry")

        if "forumdisplay.php" in url:
            forum_id = re.search(r"f=(\w+)", url).group(1)
            page = int(re.search(r"page=(\d+)", url).group(1))
            pages = self.thread_pages[forum_id]
            idx = page - 1
            has_next = idx + 1 < len(pages)
            return FakeResponse(_thread_list_html(pages[idx], has_next))

        if "showthread.php" in url:
            thread_id = re.search(r"t=(\w+)", url).group(1)
            page = int(re.search(r"page=(\d+)", url).group(1))
            pages = self.post_pages.get(thread_id, [[]])
            idx = page - 1
            has_next = idx + 1 < len(pages)
            return FakeResponse(_post_page_html(pages[idx], has_next))

        raise ValueError(f"unexpected url: {url}")


class RepeatingPostsClient:
    """Simulates the real confirmed bug: a thread whose pagination signal
    (`has_next`) claims a next page exists forever, but the actual post
    content returned never changes — seen on djelfa.info for single-page
    threads where a `rel="next"` unrelated to real post pagination (or
    the site clamping out-of-range page requests) makes `has_next` lie.
    """

    def __init__(self, posts: list[tuple]):
        self.posts = posts
        self.calls: list[str] = []

    def get(self, url: str) -> FakeResponse:
        self.calls.append(url)
        return FakeResponse(_post_page_html(self.posts, has_next=True))


class EverNewPostsClient:
    """Simulates a pathological thread that always has a next page AND
    always yields genuinely new posts — tests the hard MAX_POST_PAGES
    safety net independently of the new-posts guard (which wouldn't
    trigger here, since every page's content really is new).
    """

    def __init__(self):
        self.calls: list[str] = []

    def get(self, url: str) -> FakeResponse:
        self.calls.append(url)
        page = int(re.search(r"page=(\d+)", url).group(1))
        return FakeResponse(_post_page_html([(str(9000 + page), f"post number {page}")], has_next=True))


class ScrapeSubforumTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.state_path = Path(self.tmpdir.name) / "state.json"
        self.raw_dir = Path(self.tmpdir.name) / "raw"
        self.state = State(self.state_path)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_scrapes_all_threads_and_all_post_pages(self):
        client = FakeForumClient(
            thread_pages={"50": [[("t1", "Thread One"), ("t2", "Thread Two")]]},
            post_pages={
                "t1": [[("1001", "hello"), ("1002", "world")]],
                "t2": [[("1003", "page one")], [("1004", "page two")]],  # 2 post-pages
            },
        )
        result = scrape_subforum(client, self.state, self.raw_dir, "50")

        self.assertEqual(result["threads_considered"], 2)
        self.assertEqual(result["threads_done"], 2)
        self.assertTrue(self.state.subforum_state("50")["completed"])

        t2_lines = (self.raw_dir / "50" / "t2.jsonl").read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(t2_lines), 2)  # both post-pages of thread t2 were fetched

    def test_max_threads_caps_and_can_resume_with_higher_cap(self):
        client = FakeForumClient(
            thread_pages={"50": [[("t1", "A"), ("t2", "B"), ("t3", "C")]]},
            post_pages={"t1": [[("1001", "x")]], "t2": [[("1002", "y")]], "t3": [[("1003", "z")]]},
        )
        first = scrape_subforum(client, self.state, self.raw_dir, "50", max_threads=2)
        self.assertEqual(first["threads_considered"], 2)
        subforum_state = self.state.subforum_state("50")
        self.assertFalse(subforum_state["completed"])
        self.assertEqual(subforum_state["capped_at"], 2)

        # Same cap -> true no-op.
        calls_before = len(client.calls)
        same_cap_result = scrape_subforum(client, self.state, self.raw_dir, "50", max_threads=2)
        self.assertEqual(len(client.calls), calls_before)
        self.assertIn("already scraped up to its cap", same_cap_result.get("note", ""))

        # Higher cap -> resumes and finishes.
        result = scrape_subforum(client, self.state, self.raw_dir, "50", max_threads=3)
        self.assertEqual(result["threads_considered"], 3)
        self.assertTrue(self.state.subforum_state("50")["completed"])

    def test_completed_subforum_is_noop_on_rerun(self):
        client = FakeForumClient(
            thread_pages={"50": [[("t1", "A")]]},
            post_pages={"t1": [[("1001", "x")]]},
        )
        scrape_subforum(client, self.state, self.raw_dir, "50")
        calls_before = len(client.calls)

        result = scrape_subforum(client, self.state, self.raw_dir, "50")
        self.assertEqual(len(client.calls), calls_before)
        self.assertEqual(result.get("note"), "already completed")

    def test_thread_already_scraped_is_skipped_without_new_requests(self):
        client = FakeForumClient(
            thread_pages={"50": [[("t1", "A"), ("t2", "B")]]},
            post_pages={"t1": [[("1001", "x")]], "t2": [[("1002", "y")]]},
        )
        # Pre-scrape t1 directly, as if scraped some other way.
        scrape_thread(client, self.state, self.raw_dir, "t1", "50", "A")
        calls_after_direct_scrape = len(client.calls)

        scrape_subforum(client, self.state, self.raw_dir, "50")

        # t1's post page must not be fetched again.
        showthread_calls = [c for c in client.calls[calls_after_direct_scrape:] if "t=t1" in c]
        self.assertEqual(showthread_calls, [])

    def test_resumes_after_session_expiry_mid_walk(self):
        client = FakeForumClient(
            thread_pages={"50": [[("t1", "A"), ("t2", "B")]]},
            post_pages={"t1": [[("1001", "x")]], "t2": [[("1002", "y")]]},
        )
        client.fail_after_calls = 2  # thread-list fetch + t1's post fetch succeed, then fail

        with self.assertRaises(SessionExpiredError):
            scrape_subforum(client, self.state, self.raw_dir, "50")

        self.assertEqual(self.state.thread_state("t1")["status"], "done")

        client2 = FakeForumClient(
            thread_pages={"50": [[("t1", "A"), ("t2", "B")]]},
            post_pages={"t1": [[("1001", "x")]], "t2": [[("1002", "y")]]},
        )
        resumed_state = State(self.state_path)
        result = scrape_subforum(client2, resumed_state, self.raw_dir, "50")

        self.assertEqual(result["threads_considered"], 2)
        self.assertTrue(resumed_state.subforum_state("50")["completed"])
        # t1 must not have been re-scraped for posts.
        self.assertFalse(any("t=t1" in c for c in client2.calls))


class ScrapeThreadDuplicationBugTests(unittest.TestCase):
    """Regression tests for a real bug found in production data: two
    threads (djelfa.info f=136/t=1800757 and f=74/t=2214165), both with
    exactly 15 real posts, were repeated 1174x and 4724x respectively —
    88,440 duplicate lines out of 148,856 total raw records (~59%).
    Root cause: `has_next` claimed a next page existed forever for these
    threads, and re-fetching kept returning the same content, which the
    old code appended unconditionally on every page.
    """

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.state_path = Path(self.tmpdir.name) / "state.json"
        self.raw_dir = Path(self.tmpdir.name) / "raw"
        self.state = State(self.state_path)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_stops_when_a_page_yields_no_new_posts_despite_has_next(self):
        client = RepeatingPostsClient([("9001", "hello"), ("9002", "world")])
        status = scrape_thread(client, self.state, self.raw_dir, "t1", "50", "Some Thread")

        self.assertEqual(status, "done")
        # page 1 (genuinely new) + page 2 (confirms nothing new, then stops)
        self.assertEqual(len(client.calls), 2)

        lines = (self.raw_dir / "50" / "t1.jsonl").read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 2)  # not 2 x however many pages were fetched

    def test_max_post_pages_safety_net_stops_a_pathologically_endless_thread(self):
        client = EverNewPostsClient()
        status = scrape_thread(client, self.state, self.raw_dir, "t1", "50", "Endless Thread")

        self.assertEqual(status, "done")
        self.assertLessEqual(len(client.calls), MAX_POST_PAGES + 1)

        lines = (self.raw_dir / "50" / "t1.jsonl").read_text(encoding="utf-8").strip().splitlines()
        self.assertLessEqual(len(lines), MAX_POST_PAGES + 1)

    def test_resuming_mid_thread_does_not_reappend_already_written_posts(self):
        # Simulate a prior partial run: page 1 already written to the raw
        # file, state pointing at page 2 next.
        raw_path = self.raw_dir / "50" / "t1.jsonl"
        raw_path.parent.mkdir(parents=True)
        raw_path.write_text(
            json.dumps(
                {
                    "post_id": "9001",
                    "text": "a",
                    "thread_id": "t1",
                    "thread_title": "T",
                    "thread_url": "u",
                    "subforum_id": "50",
                    "author": "x",
                    "timestamp": "t",
                    "post_url": "p",
                    "scrape_date": "2020-01-01",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        thread_state = self.state.thread_state("t1")
        thread_state["next_post_page"] = 2
        thread_state["status"] = "pending"

        client = FakeForumClient(
            thread_pages={}, post_pages={"t1": [[("9001", "a")], [("9002", "b")]]}
        )
        scrape_thread(client, self.state, self.raw_dir, "t1", "50", "T")

        lines = raw_path.read_text(encoding="utf-8").strip().splitlines()
        post_ids = [json.loads(line)["post_id"] for line in lines]
        self.assertEqual(post_ids, ["9001", "9002"])  # no duplicate of 9001


if __name__ == "__main__":
    unittest.main()
