"""Gradio interface for the Darija n-gram text-completion demo: type a
prefix, the model samples a continuation. Same style as
Arabizi_transliteration/app.py (this project's other Gradio demo).

Uses the lightweight Python-native completion model built by
scripts/build_completion_model.py (stupid-backoff trigram over subword
tokens, counted directly from data/train.txt) -- NOT the properly
Modified-Kneser-Ney-smoothed KenLM model trained by scripts/train_ngram.py.
That one remains the "official" model behind the perplexity numbers in
plan.md section 6; this one exists specifically because generation needs
per-step next-token distributions, which kenlm's own tools (lmplz/
build_binary/query) don't expose -- they only score complete text -- and
kenlm's Python bindings (which do expose that) don't build on this
machine's Python 3.13 (see evaluate_ngram.py's docstring).
"""
from __future__ import annotations

import pickle
import random
import re
import sys
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parents[1]
N_GRAM_DIR = Path(__file__).resolve().parent
MODEL_PATH = N_GRAM_DIR / "models" / "completion_model.pkl"

sys.path.insert(0, str(ROOT / "Tokenization"))
from tokenizer_utils import load_tokenizer  # noqa: E402

tok = load_tokenizer("unigram", 20_000)

with MODEL_PATH.open("rb") as f:
    _model = pickle.load(f)

UNIGRAM_COUNTS = _model["unigram_counts"]
BIGRAM_COUNTS = _model["bigram_counts"]
TRIGRAM_COUNTS = _model["trigram_counts"]
BOS = _model["bos"]
EOS = _model["eos"]

_UNIGRAM_ITEMS = [(tok, c) for tok, c in UNIGRAM_COUNTS.items() if tok not in (BOS, EOS)]


def _sample_next(context: tuple[str, str], temperature: float) -> str:
    """Stupid backoff: trigram distribution for `context` if it exists,
    else bigram on the last token, else the overall unigram distribution.
    `temperature` reshapes the chosen distribution (counts raised to
    1/temperature before sampling) -- 1.0 is plain frequency-proportional,
    lower is more greedy/deterministic, higher is more random.
    """
    counter = TRIGRAM_COUNTS.get(context)
    if not counter:
        counter = BIGRAM_COUNTS.get(context[-1])
    if counter:
        items = list(counter.items())
    else:
        items = _UNIGRAM_ITEMS

    weights = [count ** (1.0 / temperature) for _, count in items]
    return random.choices([token for token, _ in items], weights=weights, k=1)[0]


_BYTE_FALLBACK_RE = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")


def detokenize(pieces: list[str]) -> str:
    """SentencePiece Unigram's '▁' marks word starts -- join and convert
    back to plain whitespace-separated text. Also reconstructs
    byte-fallback pieces (e.g. "<0xF0><0x9F><0x8C><0x9B>" for an emoji
    outside the trained vocabulary's normal piece coverage) back into the
    real UTF-8 character(s) they encode, instead of leaving the literal
    "<0xXX>" tokens in the output -- these can appear in generated text
    since the completion model samples from the same piece vocabulary the
    tokenizer produces, byte-fallback pieces included.
    """
    out: list[str] = []
    byte_run = bytearray()

    def flush_byte_run() -> None:
        if byte_run:
            out.append(byte_run.decode("utf-8", errors="replace"))
            byte_run.clear()

    for piece in pieces:
        match = _BYTE_FALLBACK_RE.match(piece)
        if match:
            byte_run.append(int(match.group(1), 16))
            continue
        flush_byte_run()
        out.append(piece)
    flush_byte_run()

    text = "".join(out).replace("▁", " ").strip()
    return text


def complete(prefix: str, max_new_tokens: int, temperature: float) -> str:
    prefix = (prefix or "").strip()
    prefix_pieces = tok.pieces(prefix) if prefix else []

    context_tokens = [BOS, BOS] + prefix_pieces
    generated: list[str] = []

    for _ in range(int(max_new_tokens)):
        context = (context_tokens[-2], context_tokens[-1])
        next_token = _sample_next(context, temperature)
        if next_token == EOS:
            break
        generated.append(next_token)
        context_tokens.append(next_token)

    continuation = detokenize(generated)
    if prefix and continuation:
        return f"{prefix} {continuation}"
    return prefix + continuation


description = (
    "Type a Darija text prefix (Arabic script or Arabizi) and this n-gram "
    "language model samples a continuation, trained on the augmented "
    "DarijaDZ YouTube-comment corpus (see N-gram/plan.md). This is a "
    "simple frequency-based completion model (not the smoothed KenLM "
    "model used for this project's perplexity evaluation) -- expect "
    "locally plausible but sometimes incoherent continuations, and try "
    "resubmitting for a different sample each time."
)

examples = [
    ["راني نهدر", 20, 1.0],
    ["bezzaf", 20, 1.0],
    ["اليوم الجو", 15, 0.7],
    ["salut khouya", 20, 1.0],
]

demo = gr.Interface(
    fn=complete,
    inputs=[
        gr.Textbox(label="Prefix", placeholder="Type here (e.g., راني نهدر)...", lines=3),
        gr.Slider(minimum=5, maximum=60, value=20, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.3, maximum=1.5, value=1.0, step=0.1, label="Temperature (lower = safer, higher = wilder)"),
    ],
    outputs=gr.Textbox(label="Completed text", lines=4),
    title="Darija N-gram Text Completion",
    description=description,
    examples=examples,
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch()
