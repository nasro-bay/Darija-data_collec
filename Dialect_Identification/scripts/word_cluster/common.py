"""Shared pieces for the word-cluster dialect classifier (see ../../plan.md
and README further down this docstring for the algorithm). Duplicated
`clean_for_classification`/`script_of` from label_dataset.py/build_dataset.py
rather than imported -- same reasoning as label_dataset.py's own docstring:
keeps this subpackage's dependency set (torch + sklearn, GPU venv) separate
from build_dataset.py's (datasketch, base env).

Algorithm (as specified): cluster the *word-level* embedding space (BPE
token pieces averaged per word, same convention as
Embeddings/word2vec_attention/word2vec_eval.ipynb's get_word_vector) of
the labeled training vocabulary, separately for Arabic-script words and
Latin-script words (see build_clusters.py for why separately). Each word's
occurrences across labeled training rows are tallied per dialect label
(once per row a word appears in, not per raw token occurrence -- see
build_clusters.py). Each cluster's tallies become a normalized label
probability distribution. To classify a new text: split into words, map
each word to its nearest cluster, sum that cluster's label-distribution
vector across all words in the text, and pick the argmax -- restricted to
whichever class subset the deterministic script rule allows (same
structural restriction as label_dataset.py: script=="mixed" ->
code_switch directly, script=="arabic" -> {msa, darija} only,
script=="latin" -> {arabize, french, english} only).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
DIALECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = DIALECT_DIR / "data"
MODELS_DIR = DIALECT_DIR / "models"

WORD2VEC_ATTENTION_DIR = ROOT / "Embeddings" / "word2vec_attention"
WORD2VEC_CBOW_DIR = ROOT / "Embeddings" / "word2vec" / "cbow"

sys.path.insert(0, str(ROOT / "Tokenization"))
from tokenizer_utils import load_tokenizer  # noqa: E402

# --- same regex/cleaning convention as label_dataset.py / build_dataset.py ---
_MENTION_WITH_FRAGMENT_RE = re.compile(r"\[MENTION\](\s*-[^\s]{1,10})?")
_URL_RE = re.compile(r"\[URL\]")
_INLINE_WHITESPACE_RE = re.compile(r"[ \t]+")
_ARABIC_RE = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")
_LATIN_RE = re.compile(r"[a-zA-ZÀ-ɏ]")

LATIN_CLASSES = ["arabize", "french", "english"]
ARABIC_CLASSES = ["msa", "darija"]
ALL_CONTENT_CLASSES = ARABIC_CLASSES + LATIN_CLASSES  # classes clusters ever score


def clean_for_classification(text: str) -> str:
    text = _MENTION_WITH_FRAGMENT_RE.sub("", text)
    text = _URL_RE.sub("", text)
    return _INLINE_WHITESPACE_RE.sub(" ", text).strip()


def script_of(text: str) -> str:
    has_ar = bool(_ARABIC_RE.search(text))
    has_lat = bool(_LATIN_RE.search(text))
    if has_ar and has_lat:
        return "mixed"
    if has_ar:
        return "arabic"
    if has_lat:
        return "latin"
    return "other"


def words_of(text: str) -> list[str]:
    """Whitespace word split -- same convention as the Embeddings notebooks'
    word-frequency counting (real corpus words, not BPE pieces)."""
    return text.split()


def load_embedder(variant: str = "attention"):
    """Loads the latest checkpoint of the chosen trained word2vec model
    (`attention` = CBOWAttention, well-trained to 635k+ steps; `cbow` =
    plain CBOW, far less trained -- see Embeddings/word2vec/guide.md) and
    returns a `get_word_vector(word) -> np.ndarray | None` closure, plus
    the checkpoint path used (for logging/reproducibility).

    `attention` is the default: the only one of the two with real
    evidence (see the sentiment/entity clustering ARI numbers in both
    word2vec_eval.ipynb notebooks) of separating meaningfully-different
    word groups in embedding space, which is exactly the property this
    classifier's core assumption depends on.
    """
    device = torch.device("cpu")  # embedding lookups only

    if variant == "attention":
        algo_dir = WORD2VEC_ATTENTION_DIR
        sys.path.insert(0, str(algo_dir / "scripts"))
        from model import CBOWAttention as ModelClass  # noqa: E402
        extra_ctor_kwargs = lambda ckpt_args: {  # noqa: E731
            "num_heads": ckpt_args["num_heads"], "ff_dim": ckpt_args["ff_dim"],
        }
    elif variant == "cbow":
        algo_dir = WORD2VEC_CBOW_DIR
        sys.path.insert(0, str(algo_dir / "scripts"))
        from model import CBOW as ModelClass  # noqa: E402
        extra_ctor_kwargs = lambda ckpt_args: {}  # noqa: E731
    else:
        raise ValueError(f"unknown embedder variant: {variant!r} (expected 'attention' or 'cbow')")

    data_dir = WORD2VEC_ATTENTION_DIR / "data"  # shared corpus cache either way
    models_dir = algo_dir / "models"

    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    tok = load_tokenizer(meta["tokenizer"]["key"], meta["tokenizer"]["vocab_size"])
    vocab_size = meta["tokenizer"]["vocab_size_actual"]

    checkpoints = sorted(models_dir.glob("checkpoint_step*.pt"), key=lambda p: int(p.stem.split("step")[-1]))
    if not checkpoints:
        raise SystemExit(f"No checkpoint_step*.pt found in {models_dir} -- train {variant} first.")
    checkpoint_path = checkpoints[-1]

    ckpt = torch.load(checkpoint_path, map_location=device)
    ckpt_args = ckpt["args"]
    model = ModelClass(vocab_size=vocab_size, embed_dim=ckpt_args["embed_dim"], **extra_ctor_kwargs(ckpt_args)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    embedding_matrix = model.input_embeddings.weight.detach().cpu().numpy()  # (vocab_size, embed_dim)
    pad_id = 0

    def get_word_vector(word: str) -> np.ndarray | None:
        ids = [i for i in tok.encode(word) if i != pad_id and i < vocab_size]
        if not ids:
            return None
        return embedding_matrix[ids].mean(axis=0)

    return get_word_vector, checkpoint_path
