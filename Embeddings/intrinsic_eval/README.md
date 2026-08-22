# Darija Embedding Intrinsic Evaluation

Word-similarity and analogy evaluation data for a future Darija
word-embedding model (word2vec/fastText/etc. — none trained in this
project yet; this is prep for when one exists). See `plan.md` for the
full design rationale — this file is the current-state/usage summary.

Every pair and score here was picked and scored **manually** — own
Darija linguistic knowledge, cross-checked directly against the live
corpus samples (`Data/sample_youtube.jsonl`, `Data/sample_djelfa_info.jsonl`)
where noted — not mined or auto-scored. `Data/vocab_unigrams.csv`
(built 2026-08-08) was deliberately not used as a candidate source since
it's stale relative to the corpus, which has grown substantially since.

## What's here

```
data/
  word_similarity.jsonl   # {word1, word2, category, score, note} -- 228 pairs
  analogy_pairs.jsonl     # {category, pairs: [{word_a, word_b, note}, ...]} -- 111 base pairs
scripts/
  evaluate_embeddings.py  # evaluation harness (see below)
```

### `word_similarity.jsonl` — 228 pairs

| Category | Count | Score range | What it tests |
|---|---:|---|---|
| `synonym` | 41 | ~0.65-0.90 | Same/near meaning (often dialectal-vs-MSA pairs) |
| `antonym` | 30 | ~0.05-0.15 | Opposite meaning — deliberately **low**, see below |
| `cross_script` | 51 | ~0.85-0.95 | Same word, Arabic script vs. Arabizi |
| `code_switch` | 30 | ~0.80-0.90 | French loanword vs. its Darija phonetic adaptation |
| `morphological_variant` | 36 | ~0.70-0.90 | Singular/plural or masc/fem of the same lemma |
| `unrelated` | 40 | ~0.00-0.10 | Random control pairs |

**Similarity, not relatedness**: antonyms score *low*, not high. كبير/صغير
("big"/"small") are topically related but not similar — this is the
harder SimLex-999-style choice over WordSim-353-style relatedness, and
changes how many pairs are scored versus the more common "relatedness"
convention. Keep this in mind when comparing correlation numbers against
literature that uses the other convention.

### `analogy_pairs.jsonl` — 111 base pairs, 3 relation categories

Each category is a list of base `(word_a, word_b)` pairs sharing one
relation. The evaluation script combines pairs **within the same
category** pairwise into analogy questions at eval time (same approach
as the Google analogy set / BATS — you author base pairs, not every
4-tuple by hand):

- `gender` (30 pairs) — masc/fem noun or adjective pairs
- `singular_plural` (30 pairs)
- `script_transliteration` (51 pairs) — Arabic-script word paired with
  its Arabizi form (reuses `cross_script`'s word pairs). Tests whether
  "Arabic → Arabizi" is a *consistent* vector offset across many words,
  the same way word2vec famously linearizes gender/plural in English —
  not published for Darija anywhere else.

With `n` base pairs in a category, evaluation generates `n×(n−1)`
directed analogy questions (`a1:b1 :: a2:b2`, predict `b2`).

## Running the evaluation

```bash
pip install -r requirements.txt

# Smoke-test the harness itself (no real model yet) -- expect near-zero
# Spearman correlation and near-chance analogy accuracy:
python scripts/evaluate_embeddings.py --dummy-random
```

Confirmed working: `r ≈ -0.03 to -0.31` across categories (noise around
zero, as expected for random vectors), `overall analogy accuracy ≈ 0.005`
(chance level for nearest-neighbor-of-1 over ~100 candidate words).

Once a real embedding model exists, use it as a library rather than the
CLI:

```python
import sys
sys.path.insert(0, "Embeddings/intrinsic_eval/scripts")
from evaluate_embeddings import evaluate_similarity, evaluate_analogies

def embed(word: str):
    return my_model[word] if word in my_model else None  # -> np.ndarray | None

sim_result = evaluate_similarity(embed)
analogy_result = evaluate_analogies(embed, vocab=list(my_model.key_to_index))
```

`evaluate_similarity()` returns overall + per-category Spearman
correlation, p-value, pair count, and OOV count (OOV rate is itself a
diagnostic — a model that's never seen `bezzaf` or `wallah` says
something about training-data coverage). `evaluate_analogies()` returns
overall + per-category top-1 nearest-neighbor accuracy; pass your
model's full vocabulary as `vocab` for a meaningful accuracy number —
the default (only words appearing in the analogy file) is fine for the
harness smoke test but too small a candidate pool for a real evaluation.

## Known limitations / not done yet

- 228 + 111 = 339 pairs total, short of the ~500 originally scoped —
  built as a real, quality-first first batch rather than padded to a
  round number. Extending any category further is straightforward
  (same manual-lookup process) if more coverage is wanted.
- City/region and cultural analogies were considered and dropped for
  this version — not well enough attested to mine reliably by hand from
  the corpus samples available; a possible later extension.
- No inter-annotator agreement — single-annotator scores (this project's
  explicit choice for a first version of a niche-language benchmark).
