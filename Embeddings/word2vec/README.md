# Classic CBOW + Skip-gram Word Embeddings

Two standard, non-attention word2vec baselines -- **CBOW** and
**Skip-gram** -- meant to be compared against
[`Embeddings/word2vec_attention/`](../word2vec_attention) (the CBOW +
self-attention variant) on the same corpus, tokenizer, and
hyperparameters: 128-dimensional vectors, negative sampling, frequent-
word subsampling, BPE-20K tokenizer, radius-8 context window.

Both read the prepared corpus cache directly from
`../word2vec_attention/data/` rather than duplicating a preprocessing
pass -- see `plan.md` (gitignored, internal) for the full design
rationale, and `guide.md` (gitignored, internal) for run commands.

## What's here

```
common/data_utils.py   # shared: corpus loading, negative-sampling table,
                        # subsampling probabilities, tokenizer loading,
                        # Arabizi-transliteration augmentation
cbow/scripts/           # CBOW: context words mean-pooled -> predict center
skip-gram/scripts/      # Skip-gram: center word -> predict each context word
```

## Status

Scripts implemented; training not yet run (waiting on a corpus regrow in
progress -- see `plan.md`).
