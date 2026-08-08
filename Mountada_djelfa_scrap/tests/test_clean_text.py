"""Unit tests for clean_text.py's tachkil (Arabic diacritics) stripping.

Added as a permanent regression test after a real bug was found while
writing TACHKIL_RE: typing the raw diacritic-range Unicode characters
directly into the regex got silently scrambled (RTL-aware text handling)
into a wider, wrong range that would have deleted Arabic-Indic digits
(٠-٩) along with the diacritics. Caught by direct codepoint verification
before it ever reached real data — this test guards against it recurring.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from darija_forum import clean_text  # noqa: E402


class TachkilTests(unittest.TestCase):
    def test_diacritics_stripped_base_letters_kept(self):
        self.assertEqual(clean_text.strip_tachkil("السَّلَام"), "السلام")
        self.assertEqual(clean_text.strip_tachkil("شُكْرًا"), "شكرا")

    def test_ascii_digits_survive(self):
        self.assertEqual(clean_text.strip_tachkil("عندي 1500 دج"), "عندي 1500 دج")

    def test_arabic_indic_digits_survive(self):
        # Regression: an earlier (buggy) version of TACHKIL_RE's range
        # accidentally covered U+0660-U+0669 and would have deleted these.
        self.assertEqual(clean_text.strip_tachkil("عندي ١٥٠٠ دج"), "عندي ١٥٠٠ دج")

    def test_tachkil_range_excludes_arabic_indic_digits(self):
        for cp in range(0x0660, 0x066A):
            self.assertFalse(
                clean_text.TACHKIL_RE.match(chr(cp)),
                f"TACHKIL_RE must not match Arabic-Indic digit U+{cp:04X}",
            )

    def test_clean_strips_tachkil_and_preserves_content(self):
        result = clean_text.clean("السَّلَام عَلَيْكُم وَرَحْمَة الله")
        self.assertEqual(result, "السلام عليكم ورحمة الله")

    def test_tachkil_stripped_before_quote_wrapper_username_matched(self):
        # Usernames inside the quote wrapper sometimes carry diacritics
        # (e.g. "حَمزة") -- confirm the wrapper is still stripped correctly
        # regardless (\S+ matches diacritics fine either way).
        text = "اقتباس:\nالمشاركة الأصلية كتبت بواسطة حَمزة\nهذا رد حقيقي وطويل بما فيه الكفاية"
        result = clean_text.clean(text)
        self.assertNotIn("اقتباس:", result)
        self.assertNotIn("حَمزة", result)
        self.assertIn("هذا رد حقيقي وطويل بما فيه الكفاية", result)


if __name__ == "__main__":
    unittest.main()
