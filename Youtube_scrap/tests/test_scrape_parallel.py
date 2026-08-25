"""Unit tests for _scrape_one_video_core() and scrape_videos_parallel() --
no live API calls, no real multiprocessing. worker_fn/map_fn are injected
(the dispatcher's own documented testing seam) so the dispatcher's
batching/quota/state-update orchestration is exercised deterministically
and synchronously, the same way test_scrape_channel.py tests scrape_channel
with a FakeClient instead of a real YouTubeClient.
"""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from darija_corpus.scrape import _scrape_one_video_core, scrape_videos_parallel  # noqa: E402
from darija_corpus.state import QuotaExceededError, State  # noqa: E402


class FakeClient:
    """Mimics the subset of YouTubeClient used by _scrape_one_video_core.
    `pages` is a list of comment-thread pages, each a list of already
    API-shaped thread-item dicts (see _thread_item below).
    """

    def __init__(
        self, *, channel_id="UC1", channel_title="Chan", pages=None, fail_after_pages=None, not_found=False
    ):
        self.channel_id = channel_id
        self.channel_title = channel_title
        self.pages = pages if pages is not None else [[]]
        self.fail_after_pages = fail_after_pages
        self.not_found = not_found
        self.metadata_calls = 0
        self.comment_calls = 0

    def get_video_metadata(self, video_id):
        self.metadata_calls += 1
        if self.not_found:
            return None
        return {"channel_id": self.channel_id, "channel_title": self.channel_title}

    def list_comment_threads(self, video_id, page_token=None):
        idx = int(page_token) if page_token else 0
        self.comment_calls += 1
        if self.fail_after_pages is not None and self.comment_calls > self.fail_after_pages:
            raise QuotaExceededError("simulated quota exhaustion")
        items = self.pages[idx]
        next_token = str(idx + 1) if idx + 1 < len(self.pages) else None
        return {"items": items, "nextPageToken": next_token}

    def list_replies(self, parent_id, page_token=None):
        return {"items": []}


def _thread_item(comment_id: str, text: str) -> dict:
    return {
        "snippet": {
            "topLevelComment": {"id": comment_id, "snippet": {"textDisplay": text}},
            "totalReplyCount": 0,
        }
    }


class ScrapeOneVideoCoreTests(unittest.TestCase):
    def test_done_collects_all_pages(self):
        client = FakeClient(pages=[[_thread_item("c1", "hi")], [_thread_item("c2", "bye")]])
        result = _scrape_one_video_core(client, "v1", None, None)
        self.assertEqual(result["status"], "done")
        self.assertEqual([r["comment_id"] for r in result["records"]], ["c1", "c2"])
        self.assertEqual(result["channel_id"], "UC1")

    def test_video_not_found(self):
        client = FakeClient(not_found=True)
        result = _scrape_one_video_core(client, "v1", None, None)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["reason"], "video not found")

    def test_quota_cap_hit_mid_video_returns_partial_with_resume_cursor(self):
        client = FakeClient(
            pages=[[_thread_item("c1", "hi")], [_thread_item("c2", "bye")]], fail_after_pages=1
        )
        result = _scrape_one_video_core(client, "v1", "UC1", None)
        self.assertEqual(result["status"], "partial")
        self.assertEqual([r["comment_id"] for r in result["records"]], ["c1"])
        self.assertEqual(result["next_page_token"], "1")

    def test_known_channel_id_skips_metadata_call(self):
        client = FakeClient(pages=[[]])
        _scrape_one_video_core(client, "v1", "UCknown", None)
        self.assertEqual(client.metadata_calls, 0)


class ScrapeVideosParallelTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.state = State(Path(self.tmpdir.name) / "state.json")
        self.raw_dir = Path(self.tmpdir.name) / "raw"
        self.clients: dict[str, FakeClient] = {}

    def tearDown(self):
        self.tmpdir.cleanup()

    def _worker_fn(self, args: dict) -> dict:
        client = self.clients[args["video_id"]]
        result = _scrape_one_video_core(client, args["video_id"], args["channel_id"], args["resume_page_token"])
        result["units_spent"] = min(client.comment_calls + client.metadata_calls, args["unit_cap"])
        return result

    def test_processes_different_videos_and_writes_each_raw_file(self):
        self.clients = {
            "v1": FakeClient(channel_id="UC1", pages=[[_thread_item("c1", "hi")]]),
            "v2": FakeClient(channel_id="UC2", pages=[[_thread_item("c2", "yo")]]),
        }
        results = scrape_videos_parallel(
            "fake-key", self.state, self.raw_dir, ["v1", "v2"],
            workers=2, worker_fn=self._worker_fn, map_fn=map,
        )
        self.assertEqual({r["video_id"] for r in results}, {"v1", "v2"})
        self.assertTrue(all(r["status"] == "done" for r in results))
        self.assertEqual(self.state.video_state("v1")["status"], "done")
        self.assertEqual(self.state.video_state("v2")["status"], "done")
        self.assertEqual(len((self.raw_dir / "v1.jsonl").read_text(encoding="utf-8").splitlines()), 1)
        self.assertEqual(len((self.raw_dir / "v2.jsonl").read_text(encoding="utf-8").splitlines()), 1)

    def test_already_done_videos_are_skipped_without_dispatch(self):
        pre = self.state.video_state("v1")
        pre["status"] = "done"
        pre["channel_id"] = "UC1"
        self.state.save()

        self.clients = {"v2": FakeClient(channel_id="UC2", pages=[[]])}
        dispatched: list[str] = []

        def worker_fn(args):
            dispatched.append(args["video_id"])
            return self._worker_fn(args)

        results = scrape_videos_parallel(
            "fake-key", self.state, self.raw_dir, ["v1", "v2"],
            workers=2, worker_fn=worker_fn, map_fn=map,
        )
        self.assertNotIn("v1", dispatched)
        self.assertIn("v2", dispatched)
        statuses = {r["video_id"]: r["status"] for r in results}
        self.assertEqual(statuses["v1"], "done")
        self.assertEqual(statuses["v2"], "done")

    def test_quota_cap_hit_marks_video_pending_with_resume_cursor(self):
        self.clients = {
            "v1": FakeClient(
                channel_id="UC1",
                pages=[[_thread_item("c1", "hi")], [_thread_item("c2", "bye")]],
                fail_after_pages=1,
            )
        }
        results = scrape_videos_parallel(
            "fake-key", self.state, self.raw_dir, ["v1"],
            workers=1, worker_fn=self._worker_fn, map_fn=map,
        )
        self.assertEqual(results[0]["status"], "partial")
        video_state = self.state.video_state("v1")
        self.assertEqual(video_state["status"], "pending")
        self.assertEqual(video_state["next_comment_page_token"], "1")

    def test_quota_exhausted_before_dispatch_marks_remaining_as_quota_exceeded(self):
        self.state.data["quota"]["units_used"] = 10_000  # already exhausted
        self.clients = {}
        results = scrape_videos_parallel(
            "fake-key", self.state, self.raw_dir, ["v1", "v2"],
            workers=2, worker_fn=self._worker_fn, map_fn=map,
        )
        self.assertTrue(all(r["status"] == "quota_exceeded" for r in results))

    def test_records_channel_info_from_resolved_metadata(self):
        self.clients = {"v1": FakeClient(channel_id="UC9", channel_title="Some Channel", pages=[[]])}
        scrape_videos_parallel(
            "fake-key", self.state, self.raw_dir, ["v1"],
            workers=1, worker_fn=self._worker_fn, map_fn=map,
        )
        names_file = self.state.path.parent / "channel_names.json"
        self.assertTrue(names_file.exists())


if __name__ == "__main__":
    unittest.main()
