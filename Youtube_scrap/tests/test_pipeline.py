"""Unit tests for pipeline.py: cleaning + dedup + schema-building, against
hand-written fake raw comment files (offline, no live requests).
"""
import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from darija_corpus.pipeline import run_pipeline  # noqa: E402
from darija_corpus.state import State  # noqa: E402


def _write_raw_file(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _comment(comment_id, text, video_id="v1", channel="c1", source_type="top_level"):
    return {
        "comment_id": comment_id,
        "text": text,
        "video_id": video_id,
        "channel": channel,
        "scrape_date": date.today().isoformat(),
        "source_type": source_type,
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
        # workers=1: these fixtures are a handful of comments each, so
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
        # module docstring) -- an exact-duplicate comment is now retained
        # rather than dropped, but dedup_hash is still computed and stored
        # per doc so a future standalone dedup pass (dedup.py, unchanged)
        # can reuse it without recomputing.
        _write_raw_file(
            self.raw_dir / "v1.jsonl",
            [
                _comment("1001", "wach rakoum khouya chkoun jab had le khabar الجديد اليوم"),
                _comment("1002", "wach rakoum khouya chkoun jab had le khabar الجديد اليوم"),  # exact dup
                _comment("1003", "بصح هذا خبر مختلف تماما عن الأول و لا علاقة بينهما"),
            ],
        )
        result = self._run()

        self.assertEqual(result["comments_collected"], 3)
        self.assertEqual(result["comments_retained"], 3)

        batch_path = self.processed_dir / f"batch_{date.today().isoformat()}.jsonl"
        docs = [json.loads(line) for line in batch_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(docs), 3)

        doc = docs[0]
        self.assertEqual(doc["id"], "yt_v1_1001")
        self.assertEqual(doc["source"], "youtube")
        self.assertEqual(doc["video_id"], "v1")
        self.assertIsNone(doc["script"])
        self.assertIsNone(doc["darija_confidence"])
        self.assertTrue(doc["dedup_hash"])
        # The two exact-duplicate comments hash identically -- confirms the
        # hash itself is still meaningful for a later dedup pass even though
        # nothing acts on it here.
        self.assertEqual(docs[0]["dedup_hash"], docs[1]["dedup_hash"])

    def test_text_is_cleaned_before_writing(self):
        _write_raw_file(
            self.raw_dir / "v1.jsonl",
            [_comment("1001", "شكرااااا بزاف خويا @someuser https://example.com/x")],
        )
        result = self._run()

        self.assertEqual(result["comments_retained"], 1)
        batch_path = self.processed_dir / f"batch_{date.today().isoformat()}.jsonl"
        doc = json.loads(batch_path.read_text(encoding="utf-8").splitlines()[0])
        self.assertNotIn("https://", doc["text"])
        self.assertIn("[URL]", doc["text"])
        self.assertIn("[MENTION]", doc["text"])
        self.assertNotIn("شكرااااا", doc["text"])  # elongation collapsed

    def test_near_empty_after_cleaning_is_dropped_not_retained(self):
        # Emoji-only comment -> clean_text.clean() strips all emoji and
        # returns None (residual letter count below the minimum).
        _write_raw_file(self.raw_dir / "v1.jsonl", [_comment("1001", "😂😂😂")])
        result = self._run()

        self.assertEqual(result["comments_collected"], 1)
        self.assertEqual(result["comments_dropped_empty"], 1)
        self.assertEqual(result["comments_retained"], 0)

        # pipeline.py opens the batch file unconditionally (unlike djelfa's,
        # which only opens on non-empty output) -> file exists but is empty.
        batch_path = self.processed_dir / f"batch_{date.today().isoformat()}.jsonl"
        self.assertEqual(batch_path.read_text(encoding="utf-8"), "")

    def test_cleaning_collapses_punctuation_variants_to_identical_text(self):
        # Same comment, different amounts of excessive punctuation -> distinct
        # raw text, identical after normalize_punctuation collapses runs.
        # Both are retained (dedup is disabled -- see pipeline.py's module
        # docstring), but their cleaned text and dedup_hash end up identical,
        # confirming cleaning still runs consistently ahead of hashing.
        _write_raw_file(
            self.raw_dir / "v1.jsonl",
            [
                _comment("1001", "ربي يهديك بزاف بزاف والله؟؟؟؟؟؟"),
                _comment("1002", "ربي يهديك بزاف بزاف والله؟؟"),
            ],
        )
        result = self._run()

        self.assertEqual(result["comments_collected"], 2)
        self.assertEqual(result["comments_retained"], 2)

        batch_path = self.processed_dir / f"batch_{date.today().isoformat()}.jsonl"
        docs = [json.loads(line) for line in batch_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(docs[0]["text"], docs[1]["text"])
        self.assertEqual(docs[0]["dedup_hash"], docs[1]["dedup_hash"])

    def test_log_json_tracks_dropped_empty_totals(self):
        _write_raw_file(
            self.raw_dir / "v1.jsonl",
            [
                _comment("1001", "محتوى حقيقي وفريد هنا لهذا الاختبار"),
                _comment("1002", "😂😂😂"),
            ],
        )
        self._run()

        log = json.loads(self.log_path.read_text(encoding="utf-8"))
        self.assertEqual(log["cumulative"]["comments_dropped_empty"], 1)

    def test_reruns_skip_already_processed_raw_files(self):
        _write_raw_file(self.raw_dir / "v1.jsonl", [_comment("1001", "محتوى فريد للتحقق من عدم التكرار")])
        first = self._run()
        self.assertEqual(first["comments_collected"], 1)

        second = self._run()
        self.assertEqual(second["comments_collected"], 0)
        self.assertEqual(second["videos_scraped"], 0)


if __name__ == "__main__":
    unittest.main()
