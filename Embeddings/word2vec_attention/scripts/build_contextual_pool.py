#!/usr/bin/env python
"""Builds the candidate pool for app.py's "Contextual Neighbors" tab:
reservoir-samples SAMPLE_SIZE sentences from ../data/rows.jsonl, runs each
through the trained CBOWAttention model's own self-attention+FFN block
(the same block it learned during training, just applied over a whole
sentence instead of a small context window -- no new/retrained weights),
and word-pools the resulting per-BPE-token vectors into one CONTEXTUAL
vector per WORD OCCURRENCE (not per word type -- the same word gets a
different vector in each sentence it appears in, which is the whole
point: it's what lets the app show that a word's neighbors differ by
context/sense, unlike app.py's existing static nearest-neighbor tab,
where every occurrence of a word collapses to one fixed vector).

Only word occurrences whose word appears >= MIN_OCCURRENCES times in the
sample are kept -- a word seen once has no other occurrence to meaningfully
contrast against, and dropping hapaxes keeps the pool smaller (rare words
are the bulk of a word type count, per Zipf's law, but contribute little
to "does this word have different senses" browsing).

Writes ../data/contextual_pool.npz (gitignored, same cache-to-disk
convention as app.py's word_freq_top.json): parallel arrays `words`,
`sentences` (context, truncated to CONTEXT_MAX_CHARS chars for display),
and `vectors` (float16 -- halves the ~100k-sentence pool's memory
footprint; float32 would be an unnecessary precision level for nearest-
neighbor cosine similarity here).

Run via the GPU venv's Python for training-time throughput, though
CPU-only also works, just slower (see requirements.txt):

    ".../ai-gpu/Scripts/python.exe" build_contextual_pool.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

WORD2VEC_DIR = Path(__file__).resolve().parent.parent
ROOT = WORD2VEC_DIR.parents[1]
DATA_DIR = WORD2VEC_DIR / "data"
MODELS_DIR = WORD2VEC_DIR / "models"
CHECKPOINT = MODELS_DIR / "checkpoint_step675000.pt"
OUT_PATH = DATA_DIR / "contextual_pool.npz"

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "Tokenization"))
from model import CBOWAttention  # noqa: E402
from tokenizer_utils import load_tokenizer  # noqa: E402

SAMPLE_SIZE = 100_000
MIN_OCCURRENCES = 2
CONTEXT_MAX_CHARS = 200
BATCH_SIZE = 64
MAX_TOKENS = 128  # truncate outlier-long rows (e.g. long djelfa posts) -- a
# handful of these in the same batch as short YouTube comments blew up
# self-attention's O(seq^2) memory (one real run OOM'd at 7GB on a single
# batch from padding to an outlier's length); sorting by length below
# still helps padding waste even after this cap.
SEED = 42
PAD_ID = 0


def reservoir_sample_sentences(path: Path, k: int, rng: random.Random) -> list[str]:
    reservoir: list[str] = []
    seen = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            text = json.loads(line)["text"]
            if not text.strip():
                continue
            seen += 1
            if len(reservoir) < k:
                reservoir.append(text)
            else:
                j = rng.randint(0, seen - 1)
                if j < k:
                    reservoir[j] = text
            if seen % 1_000_000 == 0:
                print(f"  ...{seen:,} rows scanned")
    print(f"Reservoir-sampled {len(reservoir):,} sentences from {seen:,} candidates")
    return reservoir


def contextualize(embedder: CBOWAttention, token_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Same block as model.py's encode_context, minus the final pooling --
    see that module's docstring. Returns (batch, seq, embed_dim), one
    vector per BPE token, conditioned on the whole sentence around it."""
    x = embedder.input_embeddings(token_ids)
    attn_out, _ = embedder.self_attn(x, x, x, key_padding_mask=~attention_mask, need_weights=False)
    x = embedder.norm1(x + attn_out)
    ffn_out = embedder.ffn(x)
    x = embedder.norm2(x + ffn_out)
    return x


def pool_tokens_to_words(
    token_embeds: torch.Tensor, attention_mask: torch.Tensor, word_ids: torch.Tensor
) -> torch.Tensor:
    """Mean-pools BPE-piece vectors belonging to the same word into one
    vector per word -- same technique as app.py's get_word_vector (static
    case) and NArabizi_sent's classifier notebook (contextual case), just
    batched here. Returns (batch, max_words, embed_dim)."""
    batch, seq, embed_dim = token_embeds.shape
    max_words = int(word_ids.max().item()) + 1
    mask_f = attention_mask.unsqueeze(-1).to(token_embeds.dtype)
    masked = token_embeds * mask_f

    sums = torch.zeros(batch, max_words, embed_dim, dtype=token_embeds.dtype, device=token_embeds.device)
    sums.scatter_add_(1, word_ids.unsqueeze(-1).expand(-1, -1, embed_dim), masked)
    counts = torch.zeros(batch, max_words, dtype=token_embeds.dtype, device=token_embeds.device)
    counts.scatter_add_(1, word_ids, attention_mask.to(token_embeds.dtype))
    return sums / counts.clamp(min=1.0).unsqueeze(-1)


def encode_words(tok, text: str) -> tuple[list[int], list[int], list[str]]:
    """Returns (token_ids, word_ids, surface_words) -- same word-boundary
    bookkeeping as NArabizi_sent's encode_words(), plus the surface word
    strings themselves (needed here to label each pooled vector; the
    classifier notebook didn't need this since it only pooled to a
    sentence-level representation, never surfaced individual words).
    Truncated to MAX_TOKENS BPE pieces (see that constant's comment) --
    any word left with zero surviving pieces after truncation is dropped
    and word_ids is remapped to stay contiguous from 0.
    """
    token_ids: list[int] = []
    word_ids: list[int] = []
    surface_words: list[str] = []
    next_word_idx = 0
    for word in text.split():
        piece_ids = [i for i in tok.encode(word) if i != PAD_ID]
        if not piece_ids:
            continue
        token_ids.extend(piece_ids)
        word_ids.extend([next_word_idx] * len(piece_ids))
        surface_words.append(word)
        next_word_idx += 1

    if len(token_ids) > MAX_TOKENS:
        token_ids = token_ids[:MAX_TOKENS]
        word_ids = word_ids[:MAX_TOKENS]
        kept_word_idxs = sorted(set(word_ids))
        remap = {old: new for new, old in enumerate(kept_word_idxs)}
        word_ids = [remap[w] for w in word_ids]
        surface_words = [surface_words[i] for i in kept_word_idxs]

    return token_ids, word_ids, surface_words


def main() -> None:
    rng = random.Random(SEED)

    print("Loading tokenizer + model...")
    meta = json.loads((DATA_DIR / "meta.json").read_text(encoding="utf-8"))
    tok = load_tokenizer(meta["tokenizer"]["key"], meta["tokenizer"]["vocab_size"])
    vocab_size = meta["tokenizer"]["vocab_size_actual"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    ckpt = torch.load(CHECKPOINT, map_location=device)
    ckpt_args = ckpt["args"]
    model = CBOWAttention(
        vocab_size=vocab_size, embed_dim=ckpt_args["embed_dim"],
        num_heads=ckpt_args["num_heads"], ff_dim=ckpt_args["ff_dim"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Sampling {SAMPLE_SIZE:,} sentences from rows.jsonl (one-time, ~1-2 min)...")
    sentences = reservoir_sample_sentences(DATA_DIR / "rows.jsonl", SAMPLE_SIZE, rng)

    print("Tokenizing (word-level, BPE pieces)...")
    all_encoded = [(*encode_words(tok, s), s) for s in sentences]
    all_encoded = [(ids, wids, words, s) for ids, wids, words, s in all_encoded if ids]
    # Sort by token length before batching -- clusters similar-length rows
    # together so a batch's padding is close to its own longest row instead
    # of the whole sample's longest row, same "length-bucketing" idea as
    # dataset.py's ClusterBatchSampler uses during training.
    all_encoded.sort(key=lambda e: len(e[0]))

    all_words: list[str] = []
    all_sentences: list[str] = []
    all_vectors: list[np.ndarray] = []

    print("Encoding sentences through the model's self-attention block...")
    with torch.no_grad():
        for batch_start in range(0, len(all_encoded), BATCH_SIZE):
            encoded = all_encoded[batch_start : batch_start + BATCH_SIZE]
            if not encoded:
                continue

            max_len = max(len(ids) for ids, _, _, _ in encoded)
            token_ids = torch.full((len(encoded), max_len), PAD_ID, dtype=torch.long)
            attention_mask = torch.zeros((len(encoded), max_len), dtype=torch.bool)
            word_ids = torch.zeros((len(encoded), max_len), dtype=torch.long)
            for i, (ids, wids, _, _) in enumerate(encoded):
                n = len(ids)
                token_ids[i, :n] = torch.tensor(ids, dtype=torch.long)
                attention_mask[i, :n] = True
                word_ids[i, :n] = torch.tensor(wids, dtype=torch.long)

            token_ids = token_ids.to(device)
            attention_mask = attention_mask.to(device)
            word_ids_dev = word_ids.to(device)

            token_embeds = contextualize(model, token_ids, attention_mask)
            word_vecs = pool_tokens_to_words(token_embeds, attention_mask, word_ids_dev).cpu().numpy()

            for i, (_, _, words, sent) in enumerate(encoded):
                context = sent if len(sent) <= CONTEXT_MAX_CHARS else sent[:CONTEXT_MAX_CHARS] + "..."
                for w_idx, w in enumerate(words):
                    all_words.append(w)
                    all_sentences.append(context)
                    all_vectors.append(word_vecs[i, w_idx])

            if (batch_start // BATCH_SIZE + 1) % 50 == 0:
                done = batch_start + len(encoded)
                print(f"  ...{done:,}/{len(all_encoded):,} sentences encoded, {len(all_words):,} occurrences so far")

    print(f"Total word occurrences before frequency filter: {len(all_words):,}")

    from collections import Counter
    word_counts = Counter(all_words)
    keep_mask = [word_counts[w] >= MIN_OCCURRENCES for w in all_words]
    kept_words = [w for w, keep in zip(all_words, keep_mask) if keep]
    kept_sentences = [s for s, keep in zip(all_sentences, keep_mask) if keep]
    kept_vectors = np.stack([v for v, keep in zip(all_vectors, keep_mask) if keep]).astype(np.float16)
    print(f"Kept {len(kept_words):,} occurrences ({len(set(kept_words)):,} distinct words) "
          f"with >= {MIN_OCCURRENCES} occurrences each")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        words=np.array(kept_words, dtype=object),
        sentences=np.array(kept_sentences, dtype=object),
        vectors=kept_vectors,
    )
    print(f"Wrote {OUT_PATH} ({kept_vectors.nbytes / 1e6:.0f} MB vectors)")


if __name__ == "__main__":
    main()
