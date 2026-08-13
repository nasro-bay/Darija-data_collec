# Tokenization — Folder Guide & Usage

This folder trains and evaluates **subword tokenizers** for the Algerian Darija
corpus. Word-level tokenization is not viable (~54% hapax legomena), so the
project compares **four tokenizer variants** across **six vocab sizes**.

## Four tokenizer variants

| # | Key | Algorithm | Trained artifact |
|---|---|---|---|
| 1 | `unigram` | SentencePiece Unigram (Kudo 2018) | `models/sentencepiece/unigram_{N}.model` |
| 2 | `unigram_sr` | Same Unigram model + **Subword Regularization** at encode time | *(reuses unigram model)* |
| 3 | `wordpiece` | SentencePiece WordPiece (BERT-style) | `models/sentencepiece/wordpiece_{N}.model` |
| 4 | `bpe` | Byte-level BPE (GPT-2 / RoBERTa) | `models/bpe/bpe_{N}/tokenizer.json` |

**Vocab sizes (N):** 500, 1,000, 5,000, 10,000, 20,000, 30,000.

Subword Regularization (Kudo 2018) is **not** a separate trained model — it
samples segmentations from the Unigram model at encode time:

```python
sp.encode(text, enable_sampling=True, alpha=0.1, nbest_size=-1)
```

Legacy pre-refactor 20K models are still supported:
- `models/sentencepiece/darija_unigram.model` → treated as `unigram` @ 20K
- `models/bpe/tokenizer.json` → treated as `bpe` @ 20K

---

## Directory layout

```
Tokenization/
├── guide.md
├── PLAN.md
├── tokenizer_utils.py           # shared paths, loading, CF, timing, fertility
├── requirements.txt
├── tokenization_eval.ipynb      # interactive comparison + visualizations
├── scripts/
│   ├── build_training_corpus.py
│   ├── train_sentencepiece.py   # --model-type unigram|wordpiece --vocab-size N
│   ├── train_bpe.py             # --vocab-size N
│   ├── train_all.py             # trains all variants × all vocab sizes
│   └── evaluate_tokenizers.py   # CLI: CF, timing, fertility, round-trip → JSON
├── data/                        # gitignored
│   ├── train_corpus.txt
│   ├── heldout_docs.jsonl
│   └── eval_results.json        # written by evaluate_tokenizers.py / notebook
└── models/                      # gitignored
    ├── sentencepiece/
    │   ├── unigram_{N}.model
    │   ├── unigram_{N}.vocab
    │   └── wordpiece_{N}.model
    └── bpe/
        └── bpe_{N}/
            ├── tokenizer.json
            ├── vocab.json
            └── merges.txt
```

---

## Quick start

```bash
cd Tokenization
pip install -r requirements.txt

# 1. Flatten processed JSONL → training file (excludes held-out sample IDs)
python scripts/build_training_corpus.py

# 2. Train all tokenizers (18 model files; SR reuses unigram models)
python scripts/train_all.py

# Or train one variant/size:
python scripts/train_sentencepiece.py --model-type unigram --vocab-size 20000
python scripts/train_sentencepiece.py --model-type wordpiece --vocab-size 20000
python scripts/train_bpe.py --vocab-size 20000

# Resume / skip already-trained models:
python scripts/train_all.py --skip-existing

# 3. CLI evaluation (timing + Compression Factor + fertility)
python scripts/evaluate_tokenizers.py

# 4. Interactive notebook
jupyter notebook tokenization_eval.ipynb
```

Training all 6 vocab sizes on ~1.17M documents takes **tens of minutes**
(SentencePiece is slower than BPE). Use `--vocab-sizes 500 1000` for a quick
smoke test first.

---

## Compression Factor (CF)

Used in `evaluate_tokenizers.py` and the notebook:

$$\text{CF} = \frac{\text{total effective tokens}}{\text{total characters} + \text{total words}}$$

Per whitespace-separated word:
- Normal: effective cost = number of subword pieces
- Contains `<unk>`: effective cost = `len(word) + 1`

**Interpretation:**
- CF closer to **0** → fewer tokens / less splitting → **better compression**
- CF closer to **1** → more unknowns / more splitting → **worse compression**

---

## Using tokenizers in Python

Prefer `tokenizer_utils.load_tokenizer()` so paths and SR settings stay consistent:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path("Tokenization")))

from tokenizer_utils import load_tokenizer, compression_factor

# Deterministic Unigram @ 20K
tok = load_tokenizer("unigram", 20_000)
ids = tok.encode("راني عارف bezzaf")
print(tok.pieces("راني عارف bezzaf"))
print(tok.decode(ids))

# Subword Regularization (sampled — non-deterministic)
tok_sr = load_tokenizer("unigram_sr", 20_000)
print(tok_sr.pieces("راني عارف bezzaf"))  # may differ each call

# WordPiece @ 10K
tok_wp = load_tokenizer("wordpiece", 10_000)

# Byte-level BPE @ 5K
tok_bpe = load_tokenizer("bpe", 5_000)

cf = compression_factor("راني عارف bezzaf", tok.pieces)
print(f"CF = {cf:.4f}")
```

---

## Evaluation outputs

### `scripts/evaluate_tokenizers.py`

Writes `data/eval_results.json` with per-(tokenizer, vocab) rows:

| Field | Description |
|---|---|
| `compression_factor_mean` | Mean CF on held-out set |
| `fertility.overall_fertility` | Tokens per whitespace-word |
| `timing.encode_ms_per_doc` | Mean encode wall time |
| `timing.decode_ms_per_doc` | Mean decode wall time |
| `roundtrip_mismatches` | Count where `decode(encode(text)) != text` |

### `tokenization_eval.ipynb`

- Full sweep table for all trained models
- CF vs vocab size (line plot)
- Encode/decode/round-trip timing plots
- Fertility by script at default vocab
- CF heatmap (tokenizer × vocab)
- 4-way segmentation visualizations (colored token boxes)

---

## Regenerating after corpus updates

```bash
python scripts/build_training_corpus.py
python scripts/train_all.py --skip-existing
python scripts/evaluate_tokenizers.py
```

Only clear `pipeline` keys in scraper state files if rebuilding processed data —
never wholesale-delete scrape state (see repo `AGENTS.md`).

---

## Choosing a tokenizer

Not automated — use CF, fertility (especially Arabic vs Latin split), timing,
and segmentation plots to decide. WordPiece integrates with BERT-style stacks;
Unigram/SR with T5/mBERT-style; byte-BPE with Hugging Face `transformers` GPT
models.

Downstream LM perplexity evaluation is out of scope until pretraining exists.

See `PLAN.md` for original design rationale and verification checklist.
