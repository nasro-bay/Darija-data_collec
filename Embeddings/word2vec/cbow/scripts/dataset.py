"""Data pipeline for classic CBOW (see ../../plan.md):

- `RowDataset`: thin wrapper, same as word2vec_attention's.
- `CBOWCollator`: per batch (not precomputed once), for eligible
  arabic-script rows with >5 words, transliterates one random word with
  20% probability, THEN BPE-tokenizes, applies frequent-word
  subsampling, and expands every row into its (context, center) training
  pairs with a radius-8 window, padded/masked to a fixed width -- exactly
  the same pair-generation logic as word2vec_attention's CBOWCollator.

No length-clustered batch sampler here, unlike word2vec_attention: mean-
pooling has no attention batch-size ceiling or padding-sensitive cost, so
a plain shuffled DataLoader over rows is sufficient (see ../../plan.md's
"What's actually different" section for why).
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[3]  # Embeddings/
sys.path.insert(0, str(ROOT / "word2vec"))
from common.data_utils import PAD_ID, maybe_augment  # noqa: E402

WINDOW_RADIUS = 8
WINDOW_SIZE = WINDOW_RADIUS * 2
# Soft safety net against a pathologically large batch (e.g. a batch
# dominated by long djelfa posts) ballooning memory -- unlike
# word2vec_attention's MAX_PAIRS_PER_BATCH, this isn't working around a
# hard attention batch-size ceiling (mean-pooling has none), just cheap
# insurance.
MAX_PAIRS_PER_BATCH = 32_768


class RowDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]


class CBOWCollator:
    def __init__(self, tok, negative_table: np.ndarray, keep_prob: dict[int, float], num_negative: int = 5):
        self.tok = tok
        self.negative_table = negative_table
        self.keep_prob = keep_prob
        self.num_negative = num_negative
        self.rng = random.Random()  # intentionally unseeded -- fresh randomness per batch/epoch

    def __call__(self, batch_rows: list[dict]):
        context_list: list[list[int]] = []
        center_list: list[int] = []

        for row in batch_rows:
            text = maybe_augment(row, self.rng)
            ids = self.tok.encode(text)
            if len(ids) < 2:
                continue

            kept = [t for t in ids if self.rng.random() < self.keep_prob.get(t, 1.0)]
            if len(kept) < 2:
                kept = ids  # don't let subsampling wipe out an already-short row entirely

            n = len(kept)
            for center_pos in range(n):
                start = max(0, center_pos - WINDOW_RADIUS)
                end = min(n, center_pos + WINDOW_RADIUS + 1)
                context = kept[start:center_pos] + kept[center_pos + 1 : end]
                if not context:
                    continue
                context_list.append(context[:WINDOW_SIZE])
                center_list.append(kept[center_pos])

        if not context_list:
            return None  # caller skips empty batches

        if len(context_list) > MAX_PAIRS_PER_BATCH:
            keep = self.rng.sample(range(len(context_list)), MAX_PAIRS_PER_BATCH)
            context_list = [context_list[i] for i in keep]
            center_list = [center_list[i] for i in keep]

        batch_size = len(context_list)
        context_ids = torch.full((batch_size, WINDOW_SIZE), PAD_ID, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, WINDOW_SIZE), dtype=torch.bool)
        for i, ctx in enumerate(context_list):
            context_ids[i, : len(ctx)] = torch.tensor(ctx, dtype=torch.long)
            attention_mask[i, : len(ctx)] = True

        center_ids = torch.tensor(center_list, dtype=torch.long)
        neg_idx = np.random.randint(0, len(self.negative_table), size=(batch_size, self.num_negative))
        negative_ids = torch.from_numpy(self.negative_table[neg_idx]).long()

        return context_ids, attention_mask, center_ids, negative_ids
