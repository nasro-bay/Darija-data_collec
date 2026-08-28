#!/usr/bin/env python
"""Gradio app: type a Darija word, get its nearest neighbors from the
trained CBOW+attention model's static input embeddings (see model.py's
docstring -- only the input table is "the word vectors", same convention
as classic word2vec). Word-level throughout, not BPE-token-level: the
candidate pool is real corpus words (whitespace-split) union curated
words from Embeddings/intrinsic_eval, and a multi-token word's vector is
the mean of its BPE pieces' embeddings (get_word_vector) -- same approach
validated in word2vec_eval.ipynb.

A second tab, "Contextual Neighbors", answers a different question: not
"what other words are like this one overall" but "what's this word doing
in THIS sentence, and what other real occurrences (of this or other
words) look similar in context" -- i.e. sense disambiguation. It reads
data/contextual_pool.npz, a pool of (word, sentence, contextual vector)
triples built by scripts/build_contextual_pool.py over 100k sampled
corpus sentences via the model's own self-attention+FFN block applied to
whole sentences (frozen, same weights, no retraining) -- see that
script's docstring for why a static word vector can't answer this. Run
that script first (via the GPU venv, for throughput) if
data/contextual_pool.npz doesn't exist yet; this app only loads the
cache, it doesn't build it (CPU inference over 100k sentences here would
be slow, unlike the one-off build script which uses CUDA when available).

Run via the GPU venv's Python (see requirements.txt):

    ".../ai-gpu/Scripts/python.exe" app.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import gradio as gr
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
WORD2VEC_DIR = Path(__file__).resolve().parent
DATA_DIR = WORD2VEC_DIR / "data"
MODELS_DIR = WORD2VEC_DIR / "models"
STATIC_DIR = WORD2VEC_DIR / "static"
EVAL_DATA_DIR = ROOT / "Embeddings" / "intrinsic_eval" / "data"
CHECKPOINT = MODELS_DIR / "checkpoint_step640000.pt"

sys.path.insert(0, str(WORD2VEC_DIR / "scripts"))
sys.path.insert(0, str(ROOT / "Tokenization"))

from model import CBOWAttention  # noqa: E402
from tokenizer_utils import load_tokenizer  # noqa: E402

ARABIC_RE = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")
LATIN_RE = re.compile(r"[a-zA-ZÀ-ɏ]")

PAD_ID = 0  # verified against Tokenization/models/bpe/bpe_20000/vocab.json
TOP_N_POOL_WORDS = 100000  # frequent corpus words in the neighbor candidate pool


def script_of(text: str) -> str:
    has_ar = bool(ARABIC_RE.search(text))
    has_lat = bool(LATIN_RE.search(text))
    if has_ar and has_lat:
        return "mixed"
    if has_ar:
        return "arabic"
    if has_lat:
        return "latin"
    return "other"


def load_curated_words() -> list[str]:
    words: set[str] = set()
    with (EVAL_DATA_DIR / "word_similarity.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            words.add(row["word1"])
            words.add(row["word2"])
    with (EVAL_DATA_DIR / "analogy_pairs.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            cat = json.loads(line)
            for p in cat["pairs"]:
                words.add(p["word_a"])
                words.add(p["word_b"])
    return sorted(words)


def load_or_build_word_freq() -> list[tuple[str, int]]:
    """Top frequent whitespace-split corpus words, cached to disk -- a full
    scan of rows.jsonl (5.5M rows) takes 1-2 minutes, too slow to redo on
    every app launch. Cache holds only the trimmed top-N, not the full
    ~3.9M-word table."""
    cache_path = DATA_DIR / "word_freq_top.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        return [tuple(pair) for pair in cached]

    print("No cached word-frequency table -- scanning rows.jsonl (one-time, ~1-2 min)...")
    counts: Counter[str] = Counter()
    with (DATA_DIR / "rows.jsonl").open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            counts.update(json.loads(line)["text"].split())
            if (i + 1) % 1_000_000 == 0:
                print(f"  ...{i + 1:,} rows scanned")

    top = counts.most_common(TOP_N_POOL_WORDS * 3)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(top, ensure_ascii=False), encoding="utf-8")
    return top


print("Loading tokenizer + model...")
meta = json.loads((DATA_DIR / "meta.json").read_text(encoding="utf-8"))
tok = load_tokenizer(meta["tokenizer"]["key"], meta["tokenizer"]["vocab_size"])
vocab_size = meta["tokenizer"]["vocab_size_actual"]

device = torch.device("cpu")  # embedding lookups only -- GPU not needed for inference here
ckpt = torch.load(CHECKPOINT, map_location=device)
ckpt_args = ckpt["args"]
model = CBOWAttention(
    vocab_size=vocab_size,
    embed_dim=ckpt_args["embed_dim"],
    num_heads=ckpt_args["num_heads"],
    ff_dim=ckpt_args["ff_dim"],
).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
embedding_matrix = model.input_embeddings.weight.detach().cpu().numpy()
print(f"Model loaded: checkpoint step={ckpt['step']:,}, vocab={vocab_size:,}")


def get_word_vector(word: str) -> np.ndarray | None:
    ids = [i for i in tok.encode(word) if i != PAD_ID and i < vocab_size]
    if not ids:
        return None
    return embedding_matrix[ids].mean(axis=0)


print("Building nearest-neighbor candidate pool...")
curated_words = load_curated_words()
word_freq_top = load_or_build_word_freq()
frequent_words = [
    w for w, _ in word_freq_top if script_of(w) in ("arabic", "latin", "mixed")
][:TOP_N_POOL_WORDS]

frequent_words_set = set(frequent_words)
curated_words_deduped = [w for w in curated_words if w not in frequent_words_set]

pool_words: list[str] = []
pool_vectors: list[np.ndarray] = []
for w in frequent_words + curated_words_deduped:
    v = get_word_vector(w)
    if v is not None:
        pool_words.append(w)
        pool_vectors.append(v)

pool_matrix = np.stack(pool_vectors)
pool_norms = np.linalg.norm(pool_matrix, axis=1, keepdims=True)
pool_normed = pool_matrix / np.clip(pool_norms, 1e-9, None)
print(f"Candidate pool ready: {len(pool_words):,} words")


CONTEXTUAL_POOL_PATH = DATA_DIR / "contextual_pool.npz"
MAX_OCCURRENCES_SHOWN = 6

ctx_words: np.ndarray | None = None
ctx_sentences: np.ndarray | None = None
ctx_normed: np.ndarray | None = None
ctx_word_to_indices: dict[str, list[int]] = {}

if CONTEXTUAL_POOL_PATH.exists():
    print("Loading contextual-neighbor pool...")
    _ctx = np.load(CONTEXTUAL_POOL_PATH, allow_pickle=True)
    ctx_words = _ctx["words"]
    ctx_sentences = _ctx["sentences"]
    ctx_vectors = _ctx["vectors"].astype(np.float32)  # float16 on disk -- promote for matmul precision
    ctx_norms = np.linalg.norm(ctx_vectors, axis=1, keepdims=True)
    ctx_normed = ctx_vectors / np.clip(ctx_norms, 1e-9, None)
    for i, w in enumerate(ctx_words):
        ctx_word_to_indices.setdefault(w, []).append(i)
    print(f"Contextual pool ready: {len(ctx_words):,} occurrences, {len(ctx_word_to_indices):,} distinct words")
else:
    print(
        f"No contextual pool found at {CONTEXTUAL_POOL_PATH} -- the "
        '"Contextual Neighbors" tab will show a message instead of results. '
        "Run scripts/build_contextual_pool.py (GPU venv) to build it."
    )


def contextual_neighbors(word: str, k: int) -> list[tuple[str, str, list[tuple[str, str, float]]]] | None:
    """Returns up to MAX_OCCURRENCES_SHOWN randomly-picked real occurrences
    of `word` (one entry per occurrence: its own sentence context, plus
    its top-k nearest-neighbor occurrences by cosine similarity over
    contextualized vectors) -- or None if the pool isn't loaded, or an
    empty list if `word` has no occurrences in the sampled pool (it's a
    100k-sentence sample, not the full corpus, so a real but rare word
    can legitimately be absent).
    """
    if ctx_normed is None:
        return None
    word = word.strip()
    indices = ctx_word_to_indices.get(word, [])
    if not indices:
        return []

    chosen = indices if len(indices) <= MAX_OCCURRENCES_SHOWN else list(
        np.random.choice(indices, size=MAX_OCCURRENCES_SHOWN, replace=False)
    )

    results = []
    for idx in chosen:
        qv = ctx_normed[idx]
        sims = ctx_normed @ qv
        order = np.argsort(-sims)
        neighbors: list[tuple[str, str, float]] = []
        for j in order:
            if j == idx:  # this exact occurrence -- always its own top match, not informative
                continue
            neighbors.append((str(ctx_words[j]), str(ctx_sentences[j]), float(sims[j])))
            if len(neighbors) >= k:
                break
        results.append((word, str(ctx_sentences[idx]), neighbors))
    return results


def nearest_neighbors(word: str, k: int) -> list[tuple[str, float]] | None:
    word = word.strip()
    if not word:
        return None
    qv = get_word_vector(word)
    if qv is None:
        return None
    qv = qv / max(np.linalg.norm(qv), 1e-9)
    sims = pool_normed @ qv
    order = np.argsort(-sims)
    results: list[tuple[str, float]] = []
    for i in order:
        w = pool_words[i]
        if w == word:
            continue
        results.append((w, float(sims[i])))
        if len(results) >= k:
            break
    return results


def render_results(word: str, k: int) -> str:
    if not word or not word.strip():
        return '<div class="empty-state">Type a word above and press Search.</div>'

    neighbors = nearest_neighbors(word, int(k))
    if neighbors is None:
        return (
            f'<div class="nn-error">No embedding found for "{word}" -- '
            f"it may be entirely outside the model's vocabulary.</div>"
        )
    if not neighbors:
        return '<div class="empty-state">No neighbors found.</div>'

    max_score = max(s for _, s in neighbors)
    cards = []
    for rank, (w, score) in enumerate(neighbors, start=1):
        pct = max(0.0, score / max_score) * 100 if max_score > 0 else 0.0
        cards.append(
            f'<div class="entry-card">'
            f'<div class="words"><span class="rank">{rank}</span>'
            f'<span class="nn-word">{w}</span></div>'
            f'<div class="score-bar-track"><div class="score-bar-fill" '
            f'style="width:{pct:.1f}%"></div></div>'
            f'<span class="score-value">cosine similarity {score:.3f}</span>'
            f"</div>"
        )
    return f'<div class="card-grid">{"".join(cards)}</div>'


def _highlight(sentence: str, word: str) -> str:
    """Bolds the first whitespace-delimited occurrence of `word` in
    `sentence`, so the queried word is easy to spot in its own context."""
    parts = sentence.split(" ")
    for i, p in enumerate(parts):
        if p == word:
            parts[i] = f"<b>{p}</b>"
            break
    return " ".join(parts)


def render_contextual_results(word: str, k: int) -> str:
    if not word or not word.strip():
        return '<div class="empty-state">Type a word above and press Search.</div>'
    if ctx_normed is None:
        return (
            '<div class="nn-error">No contextual pool found -- run '
            "scripts/build_contextual_pool.py first (see app.py's module docstring).</div>"
        )

    occurrences = contextual_neighbors(word, int(k))
    if occurrences is None:
        return '<div class="nn-error">Contextual pool failed to load.</div>'
    if not occurrences:
        return (
            f'<div class="empty-state">"{word}" doesn\'t appear in the sampled '
            "100k-sentence pool -- it may still be a real corpus word, just not "
            "one of the sentences that got sampled. Try a more frequent word, "
            'or use the "Word Neighbors" tab for a corpus-wide (non-contextual) lookup.</div>'
        )

    blocks = []
    for query_word, sentence, neighbors in occurrences:
        neighbor_cards = "".join(
            f'<div class="ctx-neighbor">'
            f'<span class="nn-word">{nw}</span>'
            f'<span class="score-value">{score:.3f}</span>'
            f'<div class="ctx-sentence">{_highlight(nsent, nw)}</div>'
            f"</div>"
            for nw, nsent, score in neighbors
        )
        blocks.append(
            f'<div class="ctx-occurrence">'
            f'<div class="ctx-query-sentence">{_highlight(sentence, query_word)}</div>'
            f'<div class="ctx-neighbor-grid">{neighbor_cards}</div>'
            f"</div>"
        )
    return "".join(blocks)


CUSTOM_CSS = """
:root {
  --dz-green: #04663a;
  --dz-red: #c8102e;
  --dz-white: #ffffff;
  --dz-gold: #b8922f;
  --dz-cream: #f7f4ee;
  --ink: #1e2723;
  --muted: #74827b;
  --border: #e2e0d8;
}

.gradio-container {
  font-family: "Cairo", "Segoe UI", Arial, sans-serif !important;
  background: var(--dz-cream) !important;
}

#topbar {
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 20px 28px;
  background: var(--dz-white);
  border: 1px solid var(--border);
  border-radius: 4px;
  margin-bottom: 4px;
}
#topbar img.logo {
  width: 44px;
  height: 44px;
  border-radius: 50%;
  border: 1px solid var(--border);
  object-fit: cover;
}
#topbar h1 {
  margin: 0;
  font-size: 1.3rem;
  font-weight: 700;
  color: var(--ink);
}
#topbar .subtitle {
  margin: 2px 0 0;
  font-size: 0.85rem;
  color: var(--muted);
}
#topbar .flag-mark {
  margin-left: auto;
  color: var(--dz-green);
}
#flagstrip {
  height: 4px;
  display: flex;
  margin-bottom: 22px;
  border-radius: 2px;
  overflow: hidden;
}
#flagstrip span { flex: 1; display: block; }
#flagstrip span:nth-child(1) { background: var(--dz-green); }
#flagstrip span:nth-child(2) { background: var(--dz-white); border-top: 1px solid var(--border); border-bottom: 1px solid var(--border); }
#flagstrip span:nth-child(3) { background: var(--dz-red); }

#query-row input[type="text"] {
  border-radius: 6px !important;
  border: 1px solid var(--border) !important;
  font-family: "Cairo", "Segoe UI", Arial, sans-serif !important;
  font-size: 1.05rem !important;
}
#search-btn {
  background: var(--dz-green) !important;
  color: white !important;
  border: none !important;
  border-radius: 6px !important;
  font-weight: 600 !important;
}
#search-btn:hover { background: #054d2c !important; }

.card-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 10px;
  padding: 4px 0;
}
.entry-card {
  background: var(--dz-white);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 14px 16px;
}
.entry-card .words {
  font-size: 1.1rem;
  font-weight: 700;
  color: var(--ink);
  display: flex;
  align-items: baseline;
  gap: 8px;
}
.entry-card .rank {
  font-size: 0.72rem;
  font-weight: 700;
  color: var(--dz-gold);
}
.entry-card .nn-word { direction: rtl; unicode-bidi: isolate; }
.entry-card .score-bar-track {
  margin-top: 10px;
  height: 5px;
  border-radius: 3px;
  background: var(--dz-cream);
  overflow: hidden;
}
.entry-card .score-bar-fill {
  height: 100%;
  background: var(--dz-green);
}
.entry-card .score-value {
  font-size: 0.72rem;
  color: var(--muted);
  margin-top: 4px;
  display: block;
}
.empty-state {
  text-align: center;
  color: var(--muted);
  padding: 40px 20px;
}
.nn-error {
  text-align: center;
  color: var(--dz-red);
  font-weight: 600;
  padding: 20px;
  background: var(--dz-white);
  border: 1px solid var(--border);
  border-radius: 4px;
}

.ctx-occurrence {
  background: var(--dz-white);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 14px 16px;
  margin-bottom: 14px;
}
.ctx-query-sentence {
  font-size: 1.05rem;
  direction: rtl;
  unicode-bidi: isolate;
  color: var(--ink);
  padding-bottom: 10px;
  margin-bottom: 10px;
  border-bottom: 1px solid var(--border);
}
.ctx-query-sentence b { color: var(--dz-green); }
.ctx-neighbor-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 8px;
}
.ctx-neighbor {
  background: var(--dz-cream);
  border-radius: 4px;
  padding: 8px 10px;
}
.ctx-neighbor .nn-word {
  font-weight: 700;
  color: var(--ink);
  direction: rtl;
  unicode-bidi: isolate;
}
.ctx-neighbor .score-value {
  float: right;
  color: var(--muted);
  font-size: 0.72rem;
}
.ctx-sentence {
  clear: both;
  margin-top: 4px;
  font-size: 0.85rem;
  color: var(--muted);
  direction: rtl;
  unicode-bidi: isolate;
}
.ctx-sentence b { color: var(--dz-green); }
"""

TOPBAR_HTML = f"""
<div id="topbar">
  <img class="logo" src="/gradio_api/file={STATIC_DIR / 'logo.png'}" alt="DarijaDZ">
  <div>
    <h1>Darija Word2Vec -- Nearest Neighbors</h1>
    <p class="subtitle">CBOW + self-attention embeddings, checkpoint step {ckpt['step']:,}</p>
  </div>
  <div class="flag-mark" aria-hidden="true">
    <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor">
      <path d="M14.5 3.5a8.5 8.5 0 1 0 0 17 8.7 8.7 0 0 1 0-17z"/>
      <path d="m17.6 8.2.9 2.7h2.9l-2.3 1.7.9 2.7-2.4-1.7-2.3 1.7.9-2.7-2.4-1.7h2.9z"/>
    </svg>
  </div>
</div>
<div id="flagstrip"><span></span><span></span><span></span></div>
"""

with gr.Blocks(title="Darija Word2Vec -- Nearest Neighbors") as demo:
    gr.HTML(TOPBAR_HTML)

    with gr.Tabs():
        with gr.Tab("Word Neighbors"):
            with gr.Row(elem_id="query-row"):
                word_input = gr.Textbox(
                    label="Word",
                    placeholder="بزاف، خدمة، مليح...",
                    scale=3,
                )
                k_slider = gr.Slider(minimum=3, maximum=25, value=10, step=1, label="Neighbors", scale=1)
                search_btn = gr.Button("Search", elem_id="search-btn", scale=1)

            results = gr.HTML('<div class="empty-state">Type a word above and press Search.</div>')

            search_btn.click(fn=render_results, inputs=[word_input, k_slider], outputs=results)
            word_input.submit(fn=render_results, inputs=[word_input, k_slider], outputs=results)

        with gr.Tab("Contextual Neighbors"):
            gr.Markdown(
                "Static neighbors above collapse every occurrence of a word to one "
                "fixed vector. This tab instead shows **real sentences** the word "
                "appeared in (sampled from the corpus) and, for each one, the "
                "nearest occurrences *in that same context* -- useful for spotting "
                "when a word means different things in different sentences."
            )
            with gr.Row(elem_id="ctx-query-row"):
                ctx_word_input = gr.Textbox(
                    label="Word",
                    placeholder="بزاف، خدمة، مليح...",
                    scale=3,
                )
                ctx_k_slider = gr.Slider(minimum=3, maximum=15, value=6, step=1, label="Neighbors per occurrence", scale=1)
                ctx_search_btn = gr.Button("Search", elem_id="search-btn", scale=1)

            ctx_results = gr.HTML('<div class="empty-state">Type a word above and press Search.</div>')

            ctx_search_btn.click(fn=render_contextual_results, inputs=[ctx_word_input, ctx_k_slider], outputs=ctx_results)
            ctx_word_input.submit(fn=render_contextual_results, inputs=[ctx_word_input, ctx_k_slider], outputs=ctx_results)


if __name__ == "__main__":
    demo.launch(css=CUSTOM_CSS, allowed_paths=[str(STATIC_DIR)])
