"""Unit tests for scrape_channel() using a fake YouTubeClient — no live
API calls. Covers: newest-first ordering, the max_videos cap, resuming
after an interruption mid-walk, a completed channel being a no-op on
rerun, and channel/video-link scrape paths sharing state so a video isn't
scraped twice.
"""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from darija_corpus.scrape import scrape_channel, scrape_video_comments  # noqa: E402
from darija_corpus.state import QuotaExceededError, State  # noqa: E402


class FakeClient:
    """Mimics the subset of YouTubeClient used by scrape_channel."""

    def __init__(self, *, channel_id: str, uploads_playlist_id: str, pages: list[list[str]]):
        self.channel_id = channel_id
        self.uploads_playlist_id = uploads_playlist_id
        self.pages = pages  # list of pages, each a list of video_ids, newest-first overall
        self.playlist_calls: list[str] = []  # page_token seen per call
        self.comment_calls: list[str] = []  # video_ids seen, in call order
        self.resolve_calls = 0
        self.fail_after_comment_calls: int | None = None  # raise QuotaExceededError after N comment calls

    def resolve_channel(self, *, channel_id=None, handle=None):
        self.resolve_calls += 1
        return {"channel_id": self.channel_id, "uploads_playlist_id": self.uploads_playlist_id}

    def list_playlist_video_ids(self, playlist_id, page_token=None):
        self.playlist_calls.append(page_token)
        idx = int(page_token) if page_token else 0
        video_ids = self.pages[idx]
        next_token = str(idx + 1) if idx + 1 < len(self.pages) else None
        return video_ids, next_token

    def list_comment_threads(self, video_id, page_token=None):
        self.comment_calls.append(video_id)
        if self.fail_after_comment_calls is not None and len(self.comment_calls) > self.fail_after_comment_calls:
            raise QuotaExceededError("simulated quota exhaustion")
        return {"items": []}

    def list_replies(self, parent_id, page_token=None):
        return {"items": []}


class ScrapeChannelTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.state_path = Path(self.tmpdir.name) / "state.json"
        self.raw_dir = Path(self.tmpdir.name) / "raw"
        self.state = State(self.state_path)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_scrapes_all_videos_newest_first(self):
        client = FakeClient(
            channel_id="UC1",
            uploads_playlist_id="UU1",
            pages=[["new3", "new2"], ["new1"]],
        )
        result = scrape_channel(client, self.state, self.raw_dir, channel_id="UC1")

        self.assertEqual(client.comment_calls, ["new3", "new2", "new1"])
        self.assertEqual(result["videos_considered"], 3)
        self.assertEqual(result["videos_done"], 3)
        self.assertTrue(self.state.channel_state("UC1")["completed"])

    def test_max_videos_caps_at_newest_n(self):
        client = FakeClient(
            channel_id="UC1",
            uploads_playlist_id="UU1",
            pages=[["new3", "new2", "new1"]],
        )
        result = scrape_channel(client, self.state, self.raw_dir, channel_id="UC1", max_videos=2)

        self.assertEqual(client.comment_calls, ["new3", "new2"])
        self.assertEqual(result["videos_considered"], 2)
        channel_state = self.state.channel_state("UC1")
        self.assertFalse(channel_state["completed"])  # capped, not fully walked
        self.assertEqual(channel_state["capped_at"], 2)

    def test_rerun_with_same_cap_is_noop(self):
        client = FakeClient(
            channel_id="UC1",
            uploads_playlist_id="UU1",
            pages=[["new3", "new2", "new1"]],
        )
        scrape_channel(client, self.state, self.raw_dir, channel_id="UC1", max_videos=2)
        calls_before, playlist_calls_before = len(client.comment_calls), len(client.playlist_calls)

        result = scrape_channel(client, self.state, self.raw_dir, channel_id="UC1", max_videos=2)

        self.assertEqual(len(client.comment_calls), calls_before)
        self.assertEqual(len(client.playlist_calls), playlist_calls_before)
        self.assertIn("already scraped up to its cap", result.get("note", ""))

    def test_raising_max_videos_resumes_a_capped_channel(self):
        client = FakeClient(
            channel_id="UC1",
            uploads_playlist_id="UU1",
            pages=[["new5", "new4", "new3"], ["new2", "new1"]],
        )
        first = scrape_channel(client, self.state, self.raw_dir, channel_id="UC1", max_videos=2)
        self.assertEqual(client.comment_calls, ["new5", "new4"])
        self.assertEqual(first["videos_considered"], 2)

        result = scrape_channel(client, self.state, self.raw_dir, channel_id="UC1", max_videos=5)

        # "new5"/"new4" re-encountered (the boundary page is re-fetched) but
        # not re-scraped; the walk continues through the rest and finishes.
        self.assertEqual(client.comment_calls, ["new5", "new4", "new3", "new2", "new1"])
        self.assertEqual(result["videos_considered"], 5)
        self.assertTrue(self.state.channel_state("UC1")["completed"])

    def test_completed_channel_is_noop_on_rerun(self):
        client = FakeClient(
            channel_id="UC1",
            uploads_playlist_id="UU1",
            pages=[["new2", "new1"]],
        )
        scrape_channel(client, self.state, self.raw_dir, channel_id="UC1")
        calls_after_first_run = len(client.comment_calls)

        result = scrape_channel(client, self.state, self.raw_dir, channel_id="UC1")

        self.assertEqual(len(client.comment_calls), calls_after_first_run)  # no new calls
        self.assertEqual(client.playlist_calls, [None])  # playlist paged only once, ever
        self.assertEqual(result.get("note"), "already completed")

    def test_resumes_after_interruption_mid_walk(self):
        # A single page ["new4", "new3"] so the crash happens *within* a
        # page, not between pages — the harder case, since resuming means
        # re-fetching that same page from its start.
        client = FakeClient(
            channel_id="UC1",
            uploads_playlist_id="UU1",
            pages=[["new4", "new3"], ["new2", "new1"]],
        )
        client.fail_after_comment_calls = 1  # succeed on "new4", fail on "new3"

        with self.assertRaises(QuotaExceededError):
            scrape_channel(client, self.state, self.raw_dir, channel_id="UC1")
        self.state.save()

        self.assertEqual(self.state.video_state("new4")["status"], "done")
        self.assertEqual(self.state.video_state("new3")["status"], "pending")  # attempted, not completed
        self.assertIn("new3", client.comment_calls)

        # Simulate a real process restart: reload state from disk rather
        # than reusing the in-memory object.
        resumed_state = State(self.state_path)
        channel_state = resumed_state.channel_state("UC1")
        self.assertFalse(channel_state["completed"])

        client2 = FakeClient(
            channel_id="UC1",
            uploads_playlist_id="UU1",
            pages=[["new4", "new3"], ["new2", "new1"]],
        )
        result = scrape_channel(client2, resumed_state, self.raw_dir, channel_id="UC1")

        # "new4" is already done, so it's re-encountered when the same page
        # is re-fetched, but must not be re-scraped or double-counted.
        self.assertNotIn("new4", client2.comment_calls)
        self.assertEqual(client2.comment_calls, ["new3", "new2", "new1"])
        self.assertTrue(resumed_state.channel_state("UC1")["completed"])
        self.assertEqual(result["videos_considered"], 4)  # not 5 — no double count

    def test_video_already_scraped_directly_is_skipped_in_channel_walk(self):
        # Simulate scrape_video() having already scraped "new2" via a direct link.
        pre_scraped = self.state.video_state("new2")
        pre_scraped["status"] = "done"
        pre_scraped["channel_id"] = "UC1"
        self.state.save()

        client = FakeClient(
            channel_id="UC1",
            uploads_playlist_id="UU1",
            pages=[["new2", "new1"]],
        )
        scrape_channel(client, self.state, self.raw_dir, channel_id="UC1")

        self.assertNotIn("new2", client.comment_calls)
        self.assertIn("new1", client.comment_calls)

    def test_channel_resolution_is_cached_across_runs(self):
        client = FakeClient(
            channel_id="UC1",
            uploads_playlist_id="UU1",
            pages=[["new1"]],
        )
        scrape_channel(client, self.state, self.raw_dir, channel_id="UC1", max_videos=1)
        self.assertEqual(client.resolve_calls, 1)

        # A fresh scrape_channel call for the same, now-completed channel
        # shouldn't re-resolve it (saves an API call every rerun).
        scrape_channel(client, self.state, self.raw_dir, channel_id="UC1", max_videos=1)
        self.assertEqual(client.resolve_calls, 1)

    def test_handle_resolution_is_cached_across_runs(self):
        client = FakeClient(
            channel_id="UC1",
            uploads_playlist_id="UU1",
            pages=[["new1"]],
        )
        scrape_channel(client, self.state, self.raw_dir, handle="@example", max_videos=1)
        self.assertEqual(client.resolve_calls, 1)

        # Same handle, fresh State reload from disk (real process restart) —
        # must resolve the channel_id from the persisted handle cache
        # instead of calling resolve_channel again.
        reloaded_state = State(self.state_path)
        scrape_channel(client, reloaded_state, self.raw_dir, handle="@example", max_videos=1)
        self.assertEqual(client.resolve_calls, 1)

    def test_scrape_video_comments_shared_helper_still_works_standalone(self):
        client = FakeClient(channel_id="UC1", uploads_playlist_id="UU1", pages=[["v1"]])
        status = scrape_video_comments(client, self.state, self.raw_dir, "v1", "UC1")
        self.assertEqual(status, "done")
        self.assertEqual(client.comment_calls, ["v1"])


if __name__ == "__main__":
    unittest.main()
