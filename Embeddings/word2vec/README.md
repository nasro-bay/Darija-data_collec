# CBOW + Self-Attention Word Embeddings

A custom word2vec variant: CBOW objective, negative sampling, but the
context-word embeddings pass through a self-attention block (+ residual
LayerNorm + FFN) before pooling and prediction, instead of a plain
average. See `plan.md` for the full design/methodology write-up.

## Run via the GPU venv

Everything here needs the GPU-enabled Python environment (the base
environment's `torch` is CPU-only):

```
GPU_PY="C:\Users\ASUS\Desktop\summer 2026\deep learning\labs\training neural networks\ai-gpu\Scripts\python.exe"
```

```bash
cd Embeddings/word2vec

# 1. Preprocess: script classification, BPE tokenization, token-frequency
#    table, K-Means (k=3) length clustering. Use --limit for a smoke test.
"$GPU_PY" scripts/build_training_data.py --limit 20000

# 2. Train
cd scripts
"$GPU_PY" train.py --epochs 1 --target-pairs-per-batch 4096
```

Confirmed working end-to-end on a 20K-doc sample: GPU active (RTX 2060),
loss dropped 3.56 → 2.39 over 300 steps.

## Key design points (see `plan.md` for full detail)

- **Embedding dim 128**, BPE-20K tokenizer (`Tokenization/models/bpe/bpe_20000`).
- **Context window**: radius 8 (up to 16 tokens), padded/masked — most
  rows are shorter than 16 tokens (short comments), so masking is the
  common case, not an edge case.
- **Frequent-word subsampling** (Mikolov's formula) + **negative sampling**
  from a unigram^0.75 table.
- **Batching**: K-Means (k=3) length clusters, one cluster per batch, to
  minimize padding waste. Rows-per-batch is derived per cluster from a
  target (context, center) pair budget (`--target-pairs-per-batch`), not
  a flat row count — a flat count let the long-document cluster generate
  far more training pairs per batch than the short one, which once blew
  past PyTorch's efficient-attention 65535-batch-size limit at full
  scale.
- **Augmentation, applied fresh per batch** (not precomputed once): for
  arabic-script rows with >5 words, 20% chance of transliterating one
  random word via `Arabizi_transliteration.transliterate_word()`.

## Not yet run at full scale

The full corpus is 5.5M+ processed docs (YouTube + djelfa forum) — the
smoke test above validates correctness on a small slice; a full training
run is a separate, much longer step.
