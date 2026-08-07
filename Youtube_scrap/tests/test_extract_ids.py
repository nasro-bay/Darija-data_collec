import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from darija_corpus.scrape import extract_channel_ref, extract_video_id  # noqa: E402


class ExtractVideoIdTests(unittest.TestCase):
    def test_watch_url(self):
        self.assertEqual(extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ"), "dQw4w9WgXcQ")

    def test_watch_url_with_extra_params(self):
        self.assertEqual(
            extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=30s"), "dQw4w9WgXcQ"
        )

    def test_youtu_be_short_link(self):
        self.assertEqual(extract_video_id("https://youtu.be/dQw4w9WgXcQ"), "dQw4w9WgXcQ")

    def test_youtu_be_with_share_query(self):
        self.assertEqual(extract_video_id("https://youtu.be/dQw4w9WgXcQ?si=abc123"), "dQw4w9WgXcQ")

    def test_shorts_url(self):
        self.assertEqual(extract_video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ"), "dQw4w9WgXcQ")

    def test_bare_id(self):
        self.assertEqual(extract_video_id("dQw4w9WgXcQ"), "dQw4w9WgXcQ")


class ExtractChannelRefTests(unittest.TestCase):
    def test_handle_url(self):
        self.assertEqual(
            extract_channel_ref("https://www.youtube.com/@Ennahartvonline/"), {"handle": "@Ennahartvonline"}
        )

    def test_handle_url_no_trailing_slash(self):
        self.assertEqual(
            extract_channel_ref("https://www.youtube.com/@Ennahartvonline"), {"handle": "@Ennahartvonline"}
        )

    def test_channel_id_url(self):
        self.assertEqual(
            extract_channel_ref("https://www.youtube.com/channel/UC57OCoLoU6zAtBdJOmwg2vA"),
            {"channel_id": "UC57OCoLoU6zAtBdJOmwg2vA"},
        )

    def test_legacy_c_url(self):
        self.assertEqual(extract_channel_ref("https://www.youtube.com/c/SomeChannel"), {"handle": "@SomeChannel"})

    def test_bare_channel_id(self):
        self.assertEqual(
            extract_channel_ref("UC57OCoLoU6zAtBdJOmwg2vA"), {"channel_id": "UC57OCoLoU6zAtBdJOmwg2vA"}
        )

    def test_bare_handle_with_at(self):
        self.assertEqual(extract_channel_ref("@example_channel"), {"handle": "@example_channel"})

    def test_bare_handle_without_at(self):
        self.assertEqual(extract_channel_ref("example_channel"), {"handle": "@example_channel"})


if __name__ == "__main__":
    unittest.main()
