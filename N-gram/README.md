# N-gram Language Model — Folder Guide & Usage

Builds a smoothed word-level (subword-tokenized) n-gram language model
over the DarijaDZ YouTube corpus, with transliteration-based augmentation
to narrow the Arabic-script/Arabizi imbalance. See `plan.md` for the full
design rationale — this file is the "how to run it" reference.

## Pipeline

```
scripts/build_augmented_corpus.py   # classify + augment -> data/augmented_corpus.jsonl
scripts/prepare_ngram_data.py       # tokenize + split -> data/train.txt, data/{dev,test}_{arabic,latin,mixed}.txt
scripts/train_ngram.py              # lmplz + build_binary -> models/darija_{trigram,4gram}.{arpa,binary}
scripts/evaluate_ngram.py           # perplexity/OOV per script bucket -> data/eval_{label}_{split}.json
```

Run in that order; each step reads the previous step's output from `data/`.

## 0. Prerequisites

- **KenLM CLI tools** (`lmplz`, `build_binary`) on `PATH` — required by
  `train_ngram.py`. These come from building kenlm's C++ project
  directly (https://github.com/kpu/kenlm, needs `cmake` + a C++
  compiler), **not** from `pip install kenlm`.
- **`kenlm` Python package** (`pip install kenlm`) — required by
  `evaluate_ngram.py` for scoring. This one *can* come from pip, but
  still needs the same C++ toolchain to build (no prebuilt Windows
  wheel as of this writing — confirmed by attempting it: pip found no
  wheel, and building from source failed with `CMAKE_C_COMPILER not
  set` / `CMAKE_CXX_COMPILER not set` until a C++ compiler — e.g. Visual
  Studio Build Tools' "Desktop development with C++" workload — is
  installed).
- The DarijaDZ pipeline's processed batches must exist:
  `Youtube_scrap/data/processed/batch_*.jsonl` (see `Youtube_scrap/youtube_scrap.md`).
- The chosen tokenizer must be trained: `Tokenization/models/sentencepiece/unigram_20000.model`
  (see `Tokenization/guide.md` — `python scripts/train_all.py` covers it).

## 1. Build the augmented corpus

```bash
cd N-gram
python scripts/build_augmented_corpus.py
```

Classifies every processed YouTube document as `arabic` / `latin` /
`mixed` script (same classification approach as
`Youtube_scrap/scripts/build_unified_dataset.py`), then:

- **Arabic-script docs**: a random 20% sample, stratified by channel, gets
  a transliterated synthetic copy added alongside the original.
- **Mixed-script docs**: **all** of them get a transliterated synthetic
  copy — `Arabizi_transliteration.transliterate()` only converts
  pure-Arabic-script word tokens and leaves Latin tokens/punctuation
  untouched by construction, so this already implements "mask the Arabic
  words, transliterate them, keep both" with no extra logic.
- **Latin-script (native Arabizi) docs**: untouched, no augmentation.

Writes `data/augmented_corpus.jsonl` — every doc (original and
synthetic) carries `script_bucket` and `is_transliterated` fields for
provenance. Nothing is ever removed or modified in place; augmentation
is strictly additive.

**Smoke test first** on a full run over ~3.7M+ docs (this calls the
transliterator per Arabic word for every augmented doc, which isn't
free): `--limit N` caps total docs read across all batch files.

```bash
python scripts/build_augmented_corpus.py --limit 5000
```

## 2. Tokenize and split

```bash
python scripts/prepare_ngram_data.py
```

Loads **SentencePiece Unigram @ 20,000 vocab** (chosen per plan.md
section 4's "read the statistics and evaluation" — see `Tokenization/data/eval_results.json`:
best compression among non-broken deterministic tokenizers at this
project's target vocab range, zero round-trip mismatches, byte-fallback
means no true subword OOV). Tokenizes every doc via
`Tokenization/tokenizer_utils.load_tokenizer("unigram", 20_000)`, then:

- Splits off up to 2,000 held-out docs **per script bucket**, half dev
  half test, drawn only from **non-synthetic** docs — plan.md section 5:
  "never evaluate on synthetic data". Configurable via
  `--heldout-per-bucket`.
- Everything else (real + synthetic) goes to `data/train.txt`.
- Writes `data/ngram_report.json`: doc counts per bucket/split, and the
  **measured** subword fertility on this corpus.

**Important deviation from plan.md's literal numbers**: section 3
assumes "~3 subword tokens per word" to justify a 9-token context window
for a word-trigram. That assumption doesn't hold for the tokenizer
actually chosen here — Unigram @ 20K measures **~1.4-1.7 tokens/word**
depending on corpus mix, not 3. This script recomputes the real subword
order needed to span a 3-word / 4-word context as `ceil(N * measured_fertility)`
and records both values in `ngram_report.json`
(`subword_order_for_word_ngram`) rather than hardcoding the plan's
literal "9" — `train_ngram.py` reads this computed value, not a
constant.

## 3. Train

```bash
python scripts/train_ngram.py
```

Trains two models via `lmplz` (Modified Kneser-Ney, kenlm's default) —
subword order read from `data/ngram_report.json`'s
`subword_order_for_word_ngram`:

- `models/darija_trigram.{arpa,binary}` — primary model, effective
  3-word context (plan.md section 5).
- `models/darija_4gram.{arpa,binary}` — comparison run, effective 4-word
  context.

Each trains on `data/train.txt`, one doc per line — `lmplz` treats each
line as an independent sequence (`<s> ... </s>`), so comment/reply
boundaries are respected without concatenating documents (plan.md
section 5).

## 4. Evaluate

```bash
python scripts/evaluate_ngram.py --label trigram --split test
python scripts/evaluate_ngram.py --label 4gram --split test
```

Reports, per script bucket (arabic / latin / mixed), on the chosen
held-out split:

- **Perplexity per word** (normalized by whitespace-word count, not
  subword-token count, so it's comparable across models/tokenizers) —
  the core check from plan.md section 6: did augmentation improve the
  Arabizi/mixed side without degrading Arabic-script performance?
- **OOV word rate** — fraction of whitespace-words containing an
  out-of-vocabulary subword piece per the *model's* training vocabulary
  (not the tokenizer's, which byte-fallback makes nearly always
  in-vocabulary at the subword level — this is measuring the LM's
  coverage, not the tokenizer's).
- **Fertility** on this specific held-out split, to sanity-check the
  window-size assumption from step 2 holds out of sample.

Writes `data/eval_{label}_{split}.json`.

## Known risks (plan.md section 7, unaddressed by this pipeline)

- No automated spot-check of transliteration quality on the augmented
  20%/mixed slice — worth a manual sample review before trusting
  augmented-side perplexity numbers.
- No content moderation / toxicity filtering pass — raw YouTube comments,
  unfiltered.
