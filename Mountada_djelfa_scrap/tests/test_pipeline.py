"""Unit tests for pipeline.py: dedup + schema-building + per-subforum
breakdown, against hand-written fake raw post files (offline, no live
requests).
"""
import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from darija_forum.pipeline import run_pipeline  # noqa: E402
from darija_forum.state import State  # noqa: E402


def _write_raw_file(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _post(post_id, text, subforum_id, thread_id="t1", thread_title="Thread"):
    return {
        "post_id": post_id,
        "text": text,
        "thread_id": thread_id,
        "thread_title": thread_title,
        "thread_url": f"https://www.djelfa.info/vb/showthread.php?t={thread_id}",
        "subforum_id": subforum_id,
        "author": "someone",
        "timestamp": "2020-01-01, 00:00",
        "post_url": f"https://www.djelfa.info/vb/showthread.php?p={post_id}#post{post_id}",
        "scrape_date": date.today().isoformat(),
    }


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.raw_dir = self.root / "raw"
        self.processed_dir = self.root / "processed"
        self.state = State(self.root / "state.json")
        self.lsh_path = self.root / "lsh.pkl"
        self.log_path = self.root / "log.json"

    def tearDown(self):
        self.tmpdir.cleanup()

    def _run(self):
        # workers=1: these fixtures are a handful of posts each, so
        # process-pool spawn overhead would dominate; single-worker keeps
        # tests fast without exercising a different code path (imap with
        # chunksize=1 over a single worker still runs through the same
        # _clean_and_hash function real runs use).
        return run_pipeline(
            raw_dir=self.raw_dir,
            processed_dir=self.processed_dir,
            state=self.state,
            lsh_path=self.lsh_path,
            log_path=self.log_path,
            workers=1,
        )

    def test_dedup_is_disabled_but_hash_still_computed(self):
        # Dedup is intentionally disabled in the pipeline (see pipeline.py's
        # module docstring) -- an exact-duplicate post is now retained
        # rather than dropped, but dedup_hash is still computed and stored
        # per doc so a future standalone dedup pass (dedup.py, unchanged)
        # can reuse it without recomputing.
        _write_raw_file(
            self.raw_dir / "50" / "t1.jsonl",
            [
                _post("1001", "wach rakoum khouya chkoun jab had le khabar الجديد اليوم", "50"),
                _post("1002", "wach rakoum khouya chkoun jab had le khabar الجديد اليوم", "50"),  # exact dup
                _post("1003", "بصح هذا خبر مختلف تماما عن الأول و لا علاقة بينهما", "50"),
            ],
        )
        result = self._run()

        self.assertEqual(result["posts_collected"], 3)
        self.assertEqual(result["posts_retained"], 3)

        batch_path = self.processed_dir / f"batch_{date.today().isoformat()}.jsonl"
        docs = [json.loads(line) for line in batch_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(docs), 3)

        doc = docs[0]
        self.assertEqual(doc["id"], "djelfa_1001")
        self.assertEqual(doc["source"], "djelfa_info")
        self.assertEqual(doc["source_type"], "forum_post")
        self.assertEqual(doc["source_metadata"]["subforum"], "50")
        self.assertEqual(doc["source_metadata"]["thread_url"], "https://www.djelfa.info/vb/showthread.php?t=t1")
        self.assertIsNone(doc["script"])
        self.assertIsNone(doc["darija_confidence"])
        self.assertTrue(doc["dedup_hash"])
        self.assertEqual(docs[0]["dedup_hash"], docs[1]["dedup_hash"])

    def test_per_subforum_breakdown(self):
        _write_raw_file(self.raw_dir / "50" / "t1.jsonl", [_post("1001", "نص فريد رقم واحد هنا", "50")])
        _write_raw_file(self.raw_dir / "77" / "t2.jsonl", [_post("2001", "نص فريد رقم اثنان هنا", "77", thread_id="t2")])

        result = self._run()

        self.assertEqual(
            result["by_subforum"]["50"],
            {"posts_collected": 1, "posts_dropped_empty": 0, "posts_retained": 1},
        )
        self.assertEqual(
            result["by_subforum"]["77"],
            {"posts_collected": 1, "posts_dropped_empty": 0, "posts_retained": 1},
        )

    def test_reruns_skip_already_processed_raw_files(self):
        _write_raw_file(self.raw_dir / "50" / "t1.jsonl", [_post("1001", "محتوى فريد للتحقق من عدم التكرار", "50")])
        first = self._run()
        self.assertEqual(first["posts_collected"], 1)

        second = self._run()
        self.assertEqual(second["posts_collected"], 0)
        self.assertEqual(second["threads_processed"], 0)

    def test_log_json_tracks_cumulative_totals(self):
        _write_raw_file(self.raw_dir / "50" / "t1.jsonl", [_post("1001", "محتوى أول للسجل التراكمي هنا", "50")])
        self._run()
        _write_raw_file(self.raw_dir / "50" / "t2.jsonl", [_post("1002", "محتوى ثاني مختلف تماما للسجل", "50", thread_id="t2")])
        self._run()

        log = json.loads(self.log_path.read_text(encoding="utf-8"))
        self.assertEqual(len(log["runs"]), 2)
        self.assertEqual(log["cumulative"]["posts_retained"], 2)
        self.assertEqual(log["cumulative_by_subforum"]["50"]["posts_retained"], 2)

    def test_text_is_cleaned_before_writing(self):
        _write_raw_file(
            self.raw_dir / "50" / "t1.jsonl",
            [_post("1001", "اقتباس:\nالمشاركة الأصلية كتبت بواسطة someone\nهذا ردي الحقيقي هنا وربي يعلم", "50")],
        )
        result = self._run()

        self.assertEqual(result["posts_retained"], 1)
        batch_path = self.processed_dir / f"batch_{date.today().isoformat()}.jsonl"
        doc = json.loads(batch_path.read_text(encoding="utf-8").splitlines()[0])
        self.assertNotIn("اقتباس:", doc["text"])
        self.assertNotIn("المشاركة الأصلية", doc["text"])
        self.assertIn("هذا ردي الحقيقي هنا وربي يعلم", doc["text"])

    def test_near_empty_after_cleaning_is_dropped_not_retained(self):
        # Pure widget-leak boilerplate -> clean_text.clean() returns None.
        _write_raw_file(
            self.raw_dir / "50" / "t1.jsonl",
            [_post("1001", "أكبر تواجد بالمنتدى كان: 25,091 بتاريخ 2019-04-29", "50")],
        )
        result = self._run()

        self.assertEqual(result["posts_collected"], 1)
        self.assertEqual(result["posts_dropped_empty"], 1)
        self.assertEqual(result["posts_retained"], 0)
        self.assertEqual(result["by_subforum"]["50"]["posts_dropped_empty"], 1)

        batch_path = self.processed_dir / f"batch_{date.today().isoformat()}.jsonl"
        self.assertFalse(batch_path.exists())

    def test_cleaning_collapses_boilerplate_variants_to_identical_text(self):
        # Same real reply, quoting two different users -> distinct raw text,
        # but identical after the quote-wrapper is stripped. Both are
        # retained (dedup is disabled -- see pipeline.py's module
        # docstring), but their cleaned text and dedup_hash end up
        # identical, confirming cleaning still runs consistently ahead of
        # hashing.
        _write_raw_file(
            self.raw_dir / "50" / "t1.jsonl",
            [
                _post("1001", "اقتباس:\nالمشاركة الأصلية كتبت بواسطة ahmed\nنعم صحيح كلامك بالضبط", "50"),
                _post("1002", "اقتباس:\nالمشاركة الأصلية كتبت بواسطة sara\nنعم صحيح كلامك بالضبط", "50"),
            ],
        )
        result = self._run()

        self.assertEqual(result["posts_collected"], 2)
        self.assertEqual(result["posts_retained"], 2)

        batch_path = self.processed_dir / f"batch_{date.today().isoformat()}.jsonl"
        docs = [json.loads(line) for line in batch_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(docs[0]["text"], docs[1]["text"])
        self.assertEqual(docs[0]["dedup_hash"], docs[1]["dedup_hash"])

    def test_log_json_tracks_dropped_empty_totals(self):
        _write_raw_file(
            self.raw_dir / "50" / "t1.jsonl",
            [
                _post("1001", "محتوى حقيقي وفريد هنا لهذا الاختبار", "50"),
                _post("1002", "أكبر تواجد بالمنتدى كان: 25,091 بتاريخ 2019-04-29", "50"),
            ],
        )
        self._run()

        log = json.loads(self.log_path.read_text(encoding="utf-8"))
        self.assertEqual(log["cumulative"]["posts_dropped_empty"], 1)
        self.assertEqual(log["cumulative_by_subforum"]["50"]["posts_dropped_empty"], 1)


if __name__ == "__main__":
    unittest.main()
