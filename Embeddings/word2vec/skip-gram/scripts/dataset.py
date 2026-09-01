"""Data pipeline for classic Skip-gram (see ../../plan.md):

- `RowDataset`: thin wrapper, same as cbow's/word2vec_attention's.
- `SkipGramCollator`: per batch (not precomputed once), same augmentation
  + BPE-tokenization + frequent-word subsampling as CBOW's collator, but
  pair generation is the one place the two algorithms structurally
  diverge: for each center token position, emits ONE (center_id,
  context_id) pair per token inside the radius-8 window (up to
  WINDOW_SIZE=16 pairs per center, vs CBOW's exactly 1 pair per center
  using the whole window at once). See ../../plan.md's "Expected
  training-time asymmetry" section -- this is why skip-gram generates
  far more training pairs per row than CBOW, a real word2vec property,
  not a bug.

No length-clustered batch sampler here either, same reasoning as cbow/.
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
from common.data_utils import maybe_augment  # noqa: E402

WINDOW_RADIUS = 8
# Soft safety net, same spirit as cbow/scripts/dataset.py's -- skip-gram
# generates far more pairs per row than CBOW, so this is sized larger to
# match (see module docstring).
MAX_PAIRS_PER_BATCH = 65_536


class RowDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]


class SkipGramCollator:
    def __init__(self, tok, negative_table: np.ndarray, keep_prob: dict[int, float], num_negative: int = 5):
        self.tok = tok
        self.negative_table = negative_table
        self.keep_prob = keep_prob
        self.num_negative = num_negative
        self.rng = random.Random()  # intentionally unseeded -- fresh randomness per batch/epoch

    def __call__(self, batch_rows: list[dict]):
        center_list: list[int] = []
        context_list: list[int] = []

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
                for ctx_pos in range(start, end):
                    if ctx_pos == center_pos:
                        continue
                    center_list.append(kept[center_pos])
                    context_list.append(kept[ctx_pos])

        if not center_list:
            return None  # caller skips empty batches

        if len(center_list) > MAX_PAIRS_PER_BATCH:
            keep = self.rng.sample(range(len(center_list)), MAX_PAIRS_PER_BATCH)
            center_list = [center_list[i] for i in keep]
            context_list = [context_list[i] for i in keep]

        batch_size = len(center_list)
        center_ids = torch.tensor(center_list, dtype=torch.long)
        context_ids = torch.tensor(context_list, dtype=torch.long)

        neg_idx = np.random.randint(0, len(self.negative_table), size=(batch_size, self.num_negative))
        negative_ids = torch.from_numpy(self.negative_table[neg_idx]).long()

        return center_ids, context_ids, negative_ids
