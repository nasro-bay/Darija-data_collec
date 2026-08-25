"""Unit tests for scrape_channel()'s parallel path (api_key given) -- no
live API calls, no real multiprocessing (worker_fn/map_fn injected, same
seam used by test_scrape_parallel.py). The sequential path (api_key
omitted, the default) is unchanged and already covered by
test_scrape_channel.py.
"""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from darija_corpus.scrape import _scrape_one_video_core, scrape_channel  # noqa: E402
from darija_corpus.state import State  # noqa: E402


class FakePlaylistClient:
    """Mimics the channel/playlist subset of YouTubeClient used by
    scrape_channel -- comment-fetching is handled separately by the
    injected worker_fn, not this client, in the parallel path.
    """

    def __init__(self, *, channel_id: str, uploads_playlist_id: str, pages: list[list[str]]):
        self.channel_id = channel_id
        self.uploads_playlist_id = uploads_playlist_id
        self.pages = pages
        self.resolve_calls = 0

    def resolve_channel(self, *, channel_id=None, handle=None):
        self.resolve_calls += 1
        return {"channel_id": self.channel_id, "uploads_playlist_id": self.uploads_playlist_id}

    def list_playlist_video_ids(self, playlist_id, page_token=None):
        idx = int(page_token) if page_token else 0
        video_ids = self.pages[idx]
        next_token = str(idx + 1) if idx + 1 < len(self.pages) else None
        return video_ids, next_token


class FakeCommentClient:
    """Per-video fake for the comment-fetching side, mirroring
    test_scrape_parallel.py's FakeClient.
    """

    def __init__(self, *, fail=False):
        self.fail = fail
        self.metadata_calls = 0
        self.comment_calls = 0

    def get_video_metadata(self, video_id):
        self.metadata_calls += 1
        return {"channel_id": "UC-should-not-be-used", "channel_title": "wrong"}

    def list_comment_threads(self, video_id, page_token=None):
        self.comment_calls += 1
        if self.fail:
            from darija_corpus.state import QuotaExceededError

            raise QuotaExceededError("simulated")
        return {"items": [], "nextPageToken": None}

    def list_replies(self, parent_id, page_token=None):
        return {"items": []}


class ScrapeChannelParallelTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.state = State(Path(self.tmpdir.name) / "state.json")
        self.raw_dir = Path(self.tmpdir.name) / "raw"
        self.comment_clients: dict[str, FakeCommentClient] = {}

    def tearDown(self):
        self.tmpdir.cleanup()

    def _worker_fn(self, args: dict) -> dict:
        client = self.comment_clients[args["video_id"]]
        result = _scrape_one_video_core(client, args["video_id"], args["channel_id"], args["resume_page_token"])
        result["units_spent"] = min(client.comment_calls + client.metadata_calls, args["unit_cap"])
        return result

    def test_parallel_path_scrapes_all_videos_and_skips_metadata_lookup(self):
        playlist_client = FakePlaylistClient(
            channel_id="UC1", uploads_playlist_id="UU1", pages=[["v2", "v1"]]
        )
        self.comment_clients = {"v1": FakeCommentClient(), "v2": FakeCommentClient()}

        result = scrape_channel(
            playlist_client, self.state, self.raw_dir, channel_id="UC1",
            api_key="fake-key", workers=2, worker_fn=self._worker_fn, map_fn=map,
        )

        self.assertEqual(result["videos_done"], 2)
        self.assertTrue(self.state.channel_state("UC1")["completed"])
        # channel_id was pre-seeded from the channel walk -- no per-video
        # metadata lookup needed (it's already known, unlike a standalone
        # scrape_video() call for an arbitrary link).
        self.assertEqual(self.comment_clients["v1"].metadata_calls, 0)
        self.assertEqual(self.comment_clients["v2"].metadata_calls, 0)
        self.assertEqual(self.state.video_state("v1")["channel_id"], "UC1")

    def test_max_videos_cap_still_respected_in_parallel_mode(self):
        playlist_client = FakePlaylistClient(
            channel_id="UC1", uploads_playlist_id="UU1", pages=[["new3", "new2", "new1"]]
        )
        self.comment_clients = {"new3": FakeCommentClient(), "new2": FakeCommentClient()}

        result = scrape_channel(
            playlist_client, self.state, self.raw_dir, channel_id="UC1", max_videos=2,
            api_key="fake-key", workers=2, worker_fn=self._worker_fn, map_fn=map,
        )

        self.assertEqual(result["videos_considered"], 2)
        self.assertNotIn("new1", self.comment_clients)  # never even dispatched
        channel_state = self.state.channel_state("UC1")
        self.assertFalse(channel_state["completed"])
        self.assertEqual(channel_state["capped_at"], 2)

    def test_quota_exhaustion_mid_page_leaves_channel_resumable(self):
        playlist_client = FakePlaylistClient(
            channel_id="UC1", uploads_playlist_id="UU1", pages=[["v1", "v2"], ["v3"]]
        )
        self.comment_clients = {}  # never dispatched -- quota is already exhausted pre-batch
        self.state.data["quota"]["units_used"] = 10_000  # fully exhausted

        result = scrape_channel(
            playlist_client, self.state, self.raw_dir, channel_id="UC1",
            api_key="fake-key", workers=2, worker_fn=self._worker_fn, map_fn=map,
        )

        self.assertTrue(result.get("quota_exceeded"))
        channel_state = self.state.channel_state("UC1")
        self.assertFalse(channel_state["completed"])
        # Page not advanced past -- a resumed run re-fetches this same page
        # (its cursor is still None, the value it started at).
        self.assertIsNone(channel_state["next_playlist_page_token"])


if __name__ == "__main__":
    unittest.main()
