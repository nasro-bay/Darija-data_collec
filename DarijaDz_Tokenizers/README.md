---
language:
  - ar

tags:
  - algerian-darija
  - tokenizer
  - nlp

library_name: tokenizers

pretty_name: DarijaDz Tokenizers

license: mit
---

# DarijaDz Tokenizers

![DarijaDz Tokenizers](Darija_dz.png)

Three subword tokenizer algorithms, trained on the same [DarijaDZ](https://huggingface.co/datasets/nasrellahkharroubi/DarijaDz)
YouTube-comment corpus, each at 5 vocabulary sizes (**1K, 5K, 10K, 20K, 30K**),
for comparison and reuse in Algerian Darija NLP / LM pretraining work.

- **SentencePiece Unigram** — script-agnostic, byte-fallback enabled, no
  whitespace-pretokenization assumption (matters since Arabic script /
  Arabizi / French mix within single documents).
- **Unigram + Subword Regularization** — the *same* Unigram model above,
  sampled at encode time (Kudo 2018 §4) instead of deterministic Viterbi
  decoding — not a separately trained artifact, see "Loading" below.
- **WordPiece** — BERT-style `##`-continuation pieces, HF `tokenizers`
  library (not SentencePiece's WordPiece mode).
- **Byte-level BPE** — GPT-2/RoBERTa-style, zero OOV by construction.

## Repo layout

```
sentencepiece/
  unigram_{1000,5000,10000,20000,30000}/
    unigram_{N}.model, unigram_{N}.vocab   # raw -- needed for Subword Regularization
    tokenizer.json                          # converted -- for AutoTokenizer (deterministic only)
    tokenizer_config.json, special_tokens_map.json
wordpiece/
  wordpiece_{1000,5000,10000,20000,30000}/
    tokenizer.json                          # self-contained (decoder embedded)
    tokenizer_config.json, special_tokens_map.json
bpe/
  bpe_{1000,5000,10000,20000,30000}/
    tokenizer.json, vocab.json, merges.txt  # self-contained (byte-level pre-tokenizer + decoder)
    tokenizer_config.json, special_tokens_map.json
```

## Loading

**Every variant works with `AutoTokenizer.from_pretrained(...)`** (point
it at any subfolder above), for deterministic encode/decode:

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("bpe/bpe_20000")
# or "wordpiece/wordpiece_20000", or "sentencepiece/unigram_20000"

ids = tok("راني عارف bezzaf")["input_ids"]
tok.decode(ids, skip_special_tokens=True)
```

WordPiece via `AutoTokenizer` still has the newline caveat below (nothing
about the HF wrapper changes that). SentencePiece Unigram via
`AutoTokenizer` is the plain deterministic model — for **Unigram + Subword
Regularization** (sampled, non-deterministic), `enable_sampling=True` only
exists in the `sentencepiece` library's own API, not in the Rust
`tokenizers`/`AutoTokenizer` port, so that variant still needs the raw
`.model` file:

```python
import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file="sentencepiece/unigram_20000/unigram_20000.model")
ids = sp.encode("راني عارف bezzaf", enable_sampling=True, alpha=0.1, nbest_size=-1)
sp.decode(ids)
```

The raw `.model`/`.vocab` files also work for plain deterministic Unigram
without `enable_sampling`, if you'd rather use the `sentencepiece` API
directly than `AutoTokenizer`.

## Known limitation: WordPiece and literal newlines

WordPiece's pre-tokenizer (`WhitespaceSplit`) treats `\n` as ordinary
whitespace, indistinguishable from a space once encoded — so
`tok.decode(tok.encode(text).ids)` on text containing a real line break
will come back with the newline collapsed to a single space. This is a
known limitation of BERT-style WordPiece tokenization generally (the
byte-level tokenizers above don't have it). If you need multi-line text to
round-trip exactly, wrap the raw tokenizer with this substitution (the
same one this project's own evaluation pipeline uses internally,
`_load_wordpiece()` in `tokenizer_utils.py`):

```python
import re
from tokenizers import Tokenizer

tok = Tokenizer.from_file("wordpiece/wordpiece_20000/tokenizer.json")
_NEWLINE_RE = re.compile(r"\s?\[NEWLINE\]\s?")

def encode(text):
    return tok.encode(text.replace("\n", " [NEWLINE] ")).ids

def decode(ids):
    decoded = tok.decode(ids, skip_special_tokens=False)
    return _NEWLINE_RE.sub("\n", decoded)
```

`[NEWLINE]` is already a registered special token in every WordPiece
vocab here, so this works with any of the 5 sizes as-is. Even with this
wrapper, one narrow edge case remains unfixed: a real space immediately
adjacent to a newline (e.g. a trailing space before a line break) can
still lose that one extra space on round-trip — full losslessness there
would require a byte-level scheme, which would make WordPiece redundant
with the BPE tokenizer above.

## Evaluation

Computed on a 1,926-document held-out set (excluded from training),
identical across all tokenizers here. **CF** (Compression Factor) — lower
is better (fewer tokens / less splitting). **Fertility** — tokens per
whitespace-word, lower is better. **Round-trip mismatches** —
`decode(encode(text)) != text` count out of 1,926.

| Tokenizer | Vocab | CF | Fertility | Round-trip mismatches |
|---|---:|---:|---:|---:|
| Unigram | 1,000 | 0.5546 | 2.7974 | 0 |
| Unigram + SR | 1,000 | 0.7349 | 3.7816 | 0 |
| WordPiece | 1,000 | 0.8105 | 4.2118 | 78 |
| BPE | 1,000 | 0.5258 | 2.6723 | 0 |
| Unigram | 5,000 | 0.3448 | 1.8159 | 0 |
| Unigram + SR | 5,000 | 0.5966 | 3.1201 | 0 |
| WordPiece | 5,000 | 0.8105 | 4.2118 | 78 |
| BPE | 5,000 | 0.4049 | 2.1276 | 0 |
| Unigram | 10,000 | 0.3048 | 1.6086 | 0 |
| Unigram + SR | 10,000 | 0.5664 | 2.9607 | 0 |
| WordPiece | 10,000 | 0.3536 | 1.8078 | 78 |
| BPE | 10,000 | 0.3687 | 1.9562 | 0 |
| Unigram | 20,000 | 0.2744 | 1.4471 | 0 |
| Unigram + SR | 20,000 | 0.5390 | 2.8310 | 0 |
| WordPiece | 20,000 | 0.2833 | 1.4535 | 78 |
| BPE | 20,000 | 0.3385 | 1.8005 | 0 |
| Unigram | 30,000 | 0.2603 | 1.3715 | 0 |
| Unigram + SR | 30,000 | 0.5292 | 2.7575 | 0 |
| WordPiece | 30,000 | 0.2629 | 1.3455 | 78 |
| BPE | 30,000 | 0.3217 | 1.7240 | 0 |

WordPiece's round-trip mismatches are exactly the 78 held-out docs
matching the space-adjacent-to-newline edge case described above — not
random noise, and stable across vocab size since it's a pre-tokenization
property, not a vocab-coverage one.

## Training data

[DarijaDZ](https://huggingface.co/datasets/nasrellahkharroubi/DarijaDz) —
~3.72M Algerian YouTube comments, cleaned (NFKC-normalized,
tachkil-stripped, near-dup filtered via MinHash/LSH). ~1.17M documents used
for training after excluding the held-out evaluation set.


## Citation

```text
Kharroubi Nasrellah.
DarijaDz Tokenizers: Algerian Darija Subword Tokenizers.
2026.
```
