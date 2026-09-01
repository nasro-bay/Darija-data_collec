# CBOW + Self-Attention Word Embeddings

A custom word2vec variant trained on this project's full corpus (YouTube
+ djelfa forum, 5.5M+ processed docs): CBOW objective + negative
sampling, but the context-word embeddings pass through a self-attention
block (+ residual, LayerNorm, feedforward) before pooling and prediction,
instead of a plain average. 128-dimensional, built on the project's own
BPE-20K tokenizer (`Tokenization/models/bpe/bpe_20000`).

Training has run well past its original scope (currently at
**step 650,000 / epoch 20**, checkpoints saved every 5,000 steps) and
can keep going indefinitely via `--resume`; the numbers below reflect
current state, not a fixed target. See `plan.md` for the full
design/methodology write-up (gitignored, internal).

## What's in this folder

```
Embeddings/word2vec/
  README.md, plan.md, guide.md   # this file / design doc / run-commands doc
                                  # (plan.md, guide.md gitignored, internal)
  requirements.txt
  word2vec_eval.ipynb            # intrinsic evaluation notebook (see below)
  app.py                         # Gradio nearest-neighbor demo (see below)
  static/logo.png                # DarijaDZ logo used in the app's topbar
  scripts/
    build_training_data.py       # preprocessing (see below)
    model.py                     # CBOWAttention nn.Module
    dataset.py                   # Dataset + length-cluster batch sampler + augmentation
    train.py                     # training loop, resumable via --resume
    build_contextual_pool.py     # precomputes app.py's contextual-neighbor cache
  data/                          # gitignored -- prepared cache + eval artifacts
  models/                        # gitignored -- checkpoints
```

## Pipeline

1. **`scripts/build_training_data.py`** -- one pass over every processed
   batch: script classification (arabic/latin/mixed), BPE tokenization,
   token-frequency counting, K-Means (k=3) length clustering. Writes
   `data/rows.jsonl` (per-row `{id, text, script_bucket, word_count,
   token_count, cluster_id}` -- raw text kept, not just token ids, since
   augmentation re-tokenizes after transliterating a word),
   `data/token_freq.json`, `data/meta.json`.
2. **`scripts/train.py`** -- trains `CBOWAttention` (`scripts/model.py`):
   context window radius 8 (≤16 tokens, masked near document
   boundaries), frequent-word subsampling (Mikolov's formula), negative
   sampling from a unigram^0.75 table, K-Means-length-clustered batching
   (one cluster per batch, rows-per-batch derived from a target
   (context, center)-pair budget so batch size stays roughly constant
   across clusters), and per-batch augmentation (20% chance per eligible
   arabic-script row of transliterating one random word via
   `Arabizi_transliteration.transliterate_word()`, applied fresh each
   epoch, not precomputed). Fully resumable (`--resume checkpoint.pt`
   restores model + optimizer state and picks up at the exact batch it
   left off at).
3. **`word2vec_eval.ipynb`** -- intrinsic evaluation: word similarity /
   analogy tasks against `Embeddings/intrinsic_eval/data/`, plus a t-SNE
   dimensionality-reduction plot of a random sample of frequent corpus
   words (currently: 300 words randomly resampled each run from the
   top-10,000 most frequent, plus a fixed curated set) for a quick visual
   sanity check of the embedding space.
4. **`app.py`** -- a Gradio demo (Algerian-flag-themed UI) with two tabs:
   - **Word Neighbors**: type a word, get its nearest neighbors from the
     model's *static* input embedding table (classic word2vec-style
     lookup -- one fixed vector per word, candidate pool = top-100K
     frequent corpus words union curated eval words).
   - **Contextual Neighbors**: shows that a word's neighbors change
     depending on the sentence it's in. Reads `data/contextual_pool.npz`
     (built by `scripts/build_contextual_pool.py`, see below) -- for a
     queried word, shows up to 6 real sampled occurrences (with sentence
     context) and, per occurrence, its own nearest-neighbor *occurrences*
     (not word types) by cosine similarity over contextualized vectors.
5. **`scripts/build_contextual_pool.py`** -- one-time precompute for the
   Contextual Neighbors tab: reservoir-samples 100,000 sentences from
   `data/rows.jsonl`, runs each through the trained model's own frozen
   self-attention+FFN block (`CBOWAttention.encode_context`'s logic,
   applied over a whole sentence and read off *before* the final
   mean-pooling step -- no retraining, same weights), word-pools the
   resulting per-BPE-token vectors into one vector per word occurrence,
   and caches `(word, sentence, vector)` triples to
   `data/contextual_pool.npz` (float16, occurrences filtered to words
   seen ≥2 times in the sample). Current cache: 1,123,113 occurrences /
   67,651 distinct words, ~288MB.

## Run via the GPU venv

Everything here needs the GPU-enabled Python environment (the base
environment's `torch` is CPU-only):

```powershell
$GPU_PY = "C:\Users\ASUS\Desktop\summer 2026\deep learning\labs\training neural networks\ai-gpu\Scripts\python.exe"
```

```powershell
cd Embeddings/word2vec
& $GPU_PY scripts\build_training_data.py          # no --limit = full corpus
cd scripts
& $GPU_PY train.py --epochs 1000 --resume ..\models\checkpoint_step650000.pt
& $GPU_PY build_contextual_pool.py                # one-time, ~10-20 min
cd ..
& $GPU_PY app.py
```

See `guide.md` (gitignored, internal) for full command-by-command detail
and flag descriptions.

## Known limitation (contextual neighbors)

The self-attention block was only ever trained on short context windows
(≤16 tokens) with its output always mean-pooled before the loss saw it
-- nothing in training pushed individual per-token outputs to stay
distinct from each other. Applying it token-by-token over a *whole*
sentence (as `build_contextual_pool.py`/app.py's Contextual Neighbors tab
do) works well for short sentences, but longer sentences can show
"neighbors" that are just other words from the same sentence (cosine
similarity ~0.99 across the board) rather than genuinely similar external
content -- a real property of the underlying model, not a bug in this
feature.
