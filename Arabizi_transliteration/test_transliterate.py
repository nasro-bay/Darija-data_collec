# test_transliterate.py
import unittest
import random
from mapping import MAPPING
from lexicon import LEXICON
from madd_rules import apply_madd_rules
from transliterate import transliterate, transliterate_word

class TestArabiziTransliterator(unittest.TestCase):
    
    def test_deterministic_seed(self):
        # Setting random seed should produce identical outputs
        random.seed(42)
        out1 = transliterate("راني عارف عليهم في الدار")
        random.seed(42)
        out2 = transliterate("راني عارف عليهم في الدار")
        self.assertEqual(out1, out2)

    def test_lexicon_hits(self):
        # "في" is in the lexicon and should always map to "fi"
        random.seed(123)
        self.assertEqual(transliterate_word("في"), "fi")
        self.assertEqual(transliterate_word("الله"), "allah")

    def test_madd_context_rules(self):
        # Test word-initial ي vs medial vs final
        # Initial: 'يبارك' -> should start with 'y'
        random.seed(42)
        out_init = transliterate_word("يبارك")
        self.assertTrue(out_init.startswith("y") or out_init.startswith("i"))
        
        # Final: 'ربي' -> should end with 'i' or 'y'
        out_final = transliterate_word("ربي")
        self.assertTrue(out_final.endswith("i") or out_final.endswith("y") or out_final.endswith("by"))
        
        # Medial: 'دير' -> should contain 'i' or 'e' between 'd' and 'r'
        out_medial = transliterate_word("دير")
        self.assertTrue("i" in out_medial or "e" in out_medial)

    def test_weight_sums(self):
        # All weights in MAPPING must sum to approximately 1.0
        for char, choices in MAPPING.items():
            weight_sum = sum(w for _, w in choices)
            self.assertAlmostEqual(weight_sum, 1.0, places=4, msg=f"Mapping weights for {char} do not sum to 1.0")

        # All weights in LEXICON must sum to approximately 1.0
        for word, choices in LEXICON.items():
            weight_sum = sum(w for _, w in choices)
            self.assertAlmostEqual(weight_sum, 1.0, places=4, msg=f"Lexicon weights for {word} do not sum to 1.0")

    def test_madd_rule_weight_sums(self):
        words_to_test = ["يبارك", "دير", "ربي", "واحد", "شكون", "باع", "انا"]
        for word in words_to_test:
            for idx, char in enumerate(word):
                if char in {'ا', 'ي', 'و'}:
                    choices = apply_madd_rules(word, idx)
                    if choices:
                        weight_sum = sum(w for _, w in choices)
                        self.assertAlmostEqual(weight_sum, 1.0, places=4, msg=f"Madd rule weights for {char} in {word} do not sum to 1.0")

if __name__ == "__main__":
    unittest.main()
