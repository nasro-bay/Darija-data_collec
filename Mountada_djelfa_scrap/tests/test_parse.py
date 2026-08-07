"""Unit tests for parse.py, against a real saved djelfa.info index page
(tests/fixtures/forum_index.html) — offline, no live requests needed.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bs4 import BeautifulSoup  # noqa: E402

from darija_forum.parse import (  # noqa: E402
    category_subforum_scope,
    extract_category_headers,
    extract_forum_links,
    looks_like_permission_denied,
)

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "forum_index.html"


class ParseRealFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        html = FIXTURE_PATH.read_text(encoding="utf-8")
        cls.soup = BeautifulSoup(html, "lxml")

    def test_finds_expected_number_of_categories(self):
        headers = extract_category_headers(self.soup)
        # Real page has ~30 genuine categories; td.tcat is also reused for
        # unrelated widgets ("آخر المواضيع", bare "#top" toggles) which
        # must NOT be counted — this pins that filtering behavior.
        self.assertGreaterEqual(len(headers), 25)
        self.assertLess(len(headers), 40)

    def test_known_top_level_categories_present(self):
        headers = dict(extract_category_headers(self.soup))
        self.assertEqual(headers.get("1"), "خيمة الجلفة")
        self.assertEqual(headers.get("9"), "منتديات الدين الإسلامي الحنيف")

    def test_finds_known_dialect_subforum_within_its_category(self):
        # منتدى اللهجة الجزائرية (Algerian Dialect forum) lives under
        # منتديات الجزائر (category forum_id=23) per the project docs.
        scope = category_subforum_scope(self.soup, "23")
        self.assertIsNotNone(scope)
        links = {link.forum_id: link.title for link in extract_forum_links(scope)}
        self.assertIn("50", links)
        self.assertEqual(links["50"], "منتدى اللهجة الجزائرية")

    def test_missing_category_scope_returns_none(self):
        self.assertIsNone(category_subforum_scope(self.soup, "999999"))

    def test_extract_forum_links_dedupes_by_forum_id(self):
        links = extract_forum_links(self.soup)
        ids = [link.forum_id for link in links]
        self.assertEqual(len(ids), len(set(ids)))

    def test_extract_forum_links_skips_empty_title_links(self):
        # Icon-only links (e.g. status-icon <a> wrapping just an <img>)
        # point at the same forum as a real text link elsewhere on the
        # page and must not create title-less/duplicate entries.
        links = extract_forum_links(self.soup)
        self.assertTrue(all(link.title for link in links))


class PermissionDeniedDetectionTests(unittest.TestCase):
    def test_flags_known_markers(self):
        self.assertTrue(looks_like_permission_denied("<html>ليس لديك الصلاحية للدخول</html>"))
        self.assertTrue(looks_like_permission_denied("<html>You do not have permission</html>"))

    def test_does_not_flag_ordinary_page(self):
        html = FIXTURE_PATH.read_text(encoding="utf-8")
        self.assertFalse(looks_like_permission_denied(html))


if __name__ == "__main__":
    unittest.main()
