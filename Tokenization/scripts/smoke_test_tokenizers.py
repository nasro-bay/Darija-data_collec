#!/usr/bin/env python
"""Quick smoke test for tokenizer_utils (run from Tokenization/)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tokenizer_utils import (
    compression_factor,
    discover_available_models,
    load_heldout_docs,
    load_tokenizer,
    time_encode_decode,
)

avail = discover_available_models()
print("available:", avail)

docs = load_heldout_docs()[:50]
texts = [d["text"] for d in docs]

for key, vs in avail:
    tok = load_tokenizer(key, vs)
    cf = sum(compression_factor(t, tok.pieces) for t in texts) / len(texts)
    rt = sum(1 for t in texts if tok.decode(tok.encode(t)) != t)
    timing = time_encode_decode(tok, texts, rounds=1)
    print(
        f"{key:12} @{vs:>6}  CF={cf:.4f}  mismatches={rt}/{len(texts)}  "
        f"enc={timing['encode_ms_per_doc']:.3f}ms"
    )

print("smoke test OK")
