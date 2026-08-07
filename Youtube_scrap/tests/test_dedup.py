import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from darija_corpus import dedup  # noqa: E402


class NearDuplicateTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.lsh_path = Path(self.tmpdir.name) / "lsh.pkl"
        self.lsh = dedup.load_lsh(self.lsh_path)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_near_identical_comments_are_flagged(self):
        original = "wach rakoum khouya chkoun jab had le score l7ala hadi bezzaf hehe"
        near_dup = "wach rakoum khouya chkoun jab had le score l7ala hadi bezzaf haha"

        first = dedup.text_to_minhash(original)
        second = dedup.text_to_minhash(near_dup)

        self.assertFalse(dedup.is_near_duplicate(self.lsh, "doc1", first))
        self.assertTrue(dedup.is_near_duplicate(self.lsh, "doc2", second))

    def test_unrelated_comments_are_not_flagged(self):
        first_text = "wallah hadi kanet mora zwina bezzaf lyoum sahbi"
        second_text = "je pense que le match va etre tres difficile ce soir"

        first = dedup.text_to_minhash(first_text)
        second = dedup.text_to_minhash(second_text)

        self.assertFalse(dedup.is_near_duplicate(self.lsh, "doc1", first))
        self.assertFalse(dedup.is_near_duplicate(self.lsh, "doc2", second))

    def test_exact_duplicate_is_flagged(self):
        text = "kayen bezzaf dyal les gens hna li ykhamou bhal hadi"
        first = dedup.text_to_minhash(text)
        second = dedup.text_to_minhash(text)

        self.assertFalse(dedup.is_near_duplicate(self.lsh, "doc1", first))
        self.assertTrue(dedup.is_near_duplicate(self.lsh, "doc2", second))

    def test_index_persists_across_reload(self):
        text = "hadi comment bech ncheki l persistence mte3 l index dyal minhash"
        mh = dedup.text_to_minhash(text)
        dedup.is_near_duplicate(self.lsh, "doc1", mh)
        dedup.save_lsh(self.lsh, self.lsh_path)

        reloaded = dedup.load_lsh(self.lsh_path)
        mh_again = dedup.text_to_minhash(text)
        self.assertTrue(dedup.is_near_duplicate(reloaded, "doc2", mh_again))


if __name__ == "__main__":
    unittest.main()
