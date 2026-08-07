"""Unit tests for parse.py's list_threads()/list_posts(), against real
saved djelfa.info pages (offline, no live requests needed).
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from darija_forum.parse import list_posts, list_threads  # noqa: E402

SUBFORUM_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "subforum_listing.html"
THREAD_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "thread_page.html"


class ListThreadsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = SUBFORUM_FIXTURE.read_text(encoding="utf-8")

    def test_finds_known_thread(self):
        threads, _ = list_threads(self.html)
        by_id = {t.thread_id: t.title for t in threads}
        self.assertEqual(by_id.get("2171447"), "ناس بكري قالوا")

    def test_dedupes_thread_ids(self):
        threads, _ = list_threads(self.html)
        ids = [t.thread_id for t in threads]
        self.assertEqual(len(ids), len(set(ids)))

    def test_detects_next_page(self):
        # Real fixture is page 1 of 73 for this subforum.
        _, has_next = list_threads(self.html)
        self.assertTrue(has_next)

    def test_no_next_page_when_no_pagenav(self):
        html_without_pagenav = "<html><body><a id='thread_title_1'>t</a></body></html>"
        _, has_next = list_threads(html_without_pagenav)
        self.assertFalse(has_next)


class ListPostsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = THREAD_FIXTURE.read_text(encoding="utf-8")

    def test_finds_known_post_with_author_and_timestamp(self):
        posts, _ = list_posts(self.html)
        by_id = {p.post_id: p for p in posts}
        post = by_id.get("3997725048")
        self.assertIsNotNone(post)
        self.assertEqual(post.author, "crono")
        self.assertEqual(post.timestamp, "2018-11-29, 11:52")
        self.assertIn("بسم الله", post.text)

    def test_post_url_is_well_formed(self):
        posts, _ = list_posts(self.html)
        post = next(p for p in posts if p.post_id == "3997725048")
        self.assertEqual(
            post.post_url, "https://www.djelfa.info/vb/showthread.php?p=3997725048#post3997725048"
        )

    def test_all_posts_have_nonempty_text(self):
        posts, _ = list_posts(self.html)
        self.assertGreater(len(posts), 0)
        self.assertTrue(all(p.text for p in posts))

    def test_detects_next_page(self):
        _, has_next = list_posts(self.html)
        self.assertTrue(has_next)  # fixture is page 1 of 5


if __name__ == "__main__":
    unittest.main()
