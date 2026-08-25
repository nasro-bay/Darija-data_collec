# Algerian Darija Text Corpus

![DarijaDZ](Darija_dz.png)

A collection pipeline and toolset for Algerian Darija (Algerian Arabic
dialect) — a low-resource language variety written in both Arabic script
and Arabizi (Latin script with numerals standing in for Arabic sounds,
e.g. `3=ع`, `7=ح`, `9=ق`), and frequently mixed with French. Darija is
spoken by tens of millions of people but remains comparatively
under-resourced next to MSA and other major languages — little
annotated data, few tokenizers or embeddings trained specifically for
it, and almost no dialect-aware tooling. This project builds toward
closing that gap: a clean corpus, a tokenizer, word embeddings, a
dialect classifier, and benchmark evaluations, each in its own
subdirectory (see each one's own `README.md` for detail — this file
stays brief on purpose).

## Data collection

| Source | Directory | Method |
|---|---|---|
| YouTube comments (Algerian channels: vlogs, comedy, sports, news, podcasts) | [`Youtube_scrap/`](Youtube_scrap) | YouTube Data API v3, curated seed channels/videos |
| Djelfa.info forum (منتديات الجلفة) | [`Mountada_djelfa_scrap/`](Mountada_djelfa_scrap) | Direct scraping, robots.txt-compliant |

Both sources go through the same shape of pipeline: scrape → clean →
schema → JSONL. Cleaning is regex-based text normalization, derived
empirically against real samples rather than assumed: collapsing
elongated letters, punctuation runs, and repeated emoji; stripping
Arabic diacritics (tachkil); anonymizing URLs and @mentions into
placeholders; and (forum-specific) stripping BBCode, quote-wrappers, and
tatweel. Near-duplicate detection (MinHash/LSH) exists as a reusable
module for later use.

## Tokenization ([`Tokenization/`](Tokenization))

Multiple tokenizer variants trained and compared head-to-head at
matched vocabulary sizes (1K–30K): SentencePiece Unigram (with and
without subword regularization), WordPiece, and byte-level BPE.
Evaluated on held-out data via fertility, compression factor,
vocabulary utilization, OOV/byte-fallback rate, known-word
fragmentation under each tokenizer, and round-trip fidelity.

## Embeddings ([`Embeddings/word2vec/`](Embeddings/word2vec))

A custom word2vec variant: CBOW objective and negative sampling, but the
context-word embeddings pass through a self-attention block (+ residual,
LayerNorm, feedforward) before pooling and prediction, instead of a
plain average. 128-dimensional, trained on the full corpus — full
training run took roughly 30 hours.

## Dialect identification ([`Dialect_Identification/`](Dialect_Identification))

**Goal**: classify Algerian online text into `french` / `arabize`
(Arabizi) / `msa` / `darija` (Arabic script) / `english` /
`code_switch`, to support corpus filtering and dialect-aware downstream
work. **Method**: a hybrid, not a pure model call — deterministic
script detection (regex) handles code-switch detection directly and
restricts which classes a local instruction model (Qwen3.5-4B, 4-bit)
is even allowed to choose from based on the text's script, closing
script-impossible mistakes (e.g. labeling pure-Latin text as
Arabic-script Darija) by construction rather than by prompting alone.

## Benchmarks ([`Benchmarks/`](Benchmarks))

External datasets used to evaluate against, not collected by this
project:

| Benchmark | Field |
|---|---|
| [NArabizi](Benchmarks/NArabizi-main) | POS tagging / dependency parsing, with explicit per-token code-switching annotation |
| [Twifil](Benchmarks/Twifil) | Sentiment analysis (North African Arabic, mixed Algerian/Moroccan/Tunisian) |

## Status

Active. See each subdirectory's own `README.md`/scripts for usage.
