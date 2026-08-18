# NArabizi — benchmark notes

Second entry in `Benchmarks/` (after `Twifil/`). A full clone of the
**NArabizi** research repo — an actual academic Algerian Arabizi
treebank with POS/dependency/sentiment/topic annotations, not a scraped
dump. Its own `README.md` was empty and its `experiments/` scripts are
paper-reproduction code (zero-shot cross-lingual transfer experiments),
not documentation — this file is the result of directly parsing the data
in `data/`, via `scripts/explore_narabizi.py` (rerun it to refresh the
numbers below; nothing here is hand-typed from memory).

## Why this one matters more than Twifil

Unlike Twifil (a loose "North African Arabic" Twitter scrape with no
dialect ground truth), this is **specifically, deliberately Algerian**:
every token in the core treebank carries an explicit `lang=` tag
(`ar_dz`, `fr`, `ar_msa`, `en`, ...), so code-switching can be measured
exactly instead of guessed at with a script-based heuristic. It's also
linguistically annotated (POS tags, dependency parses), which Twifil
isn't.

## What's actually in `data/`

Three genuinely different resources live side by side — **don't
conflate them**, they're easy to mix up by name alone:

1. **`Narabizi/`** — the core treebank: 997 train / 137 dev / 144 test
   sentences (1,278 total), real Algerian tweets in Arabizi (Latin
   transliteration), each with POS tags, a dependency parse, a French
   translation (`trad_fr`), and per-token `lang=` code-switching tags.
   `Narabizi/sentiment/` and `Narabizi/topic/` are **label-only files
   keyed by `sent_id`**, not standalone text — they must be joined
   against `Narabizi/pos/*.conllu` to get (text, label) pairs (see
   `load_label_file()` / `parse_conllu()` in the exploration script for
   exactly how).
2. **`ud/ar_dz/` and `ud/ar_na/`** — **the same 997/137/144 sentences as
   `Narabizi/pos/`**, re-exported in standard 10-column CoNLL-U (the
   `Narabizi/pos/` files use an extended, non-standard column layout with
   extra French/Arabic-script fields). Confirmed identical sentence
   counts and content — `ar_dz` and `ar_na` are **not two different
   datasets**, they're the same treebank with the `FORM` and `LEMMA`
   columns swapped (`ar_dz`: FORM=Arabic-script, LEMMA=Arabizi; `ar_na`:
   FORM=Arabizi, LEMMA=Arabic-script) — a paper-experiment convenience
   for testing which script a downstream parser should see as the primary
   form, not two distinct annotation efforts.
3. **`sentiment/`** — a **separate, unrelated sentiment resource**, not
   connected to the treebank above (different domain: football/political
   tweets, not general conversation) with its own `ar_dz`/`ar_na` pair
   (564 train / 75 dev / 92 test = 731 rows, binary 0/1 label, **also**
   the same content in both scripts — same swap pattern as #2) **plus**
   five other languages used as auxiliary/source data for the paper's
   zero-shot transfer experiments (see below) — those aren't Algerian at
   all and shouldn't be mistaken for more Darija data.

## Code-switching (measured directly, not guessed)

From the treebank's own per-token `lang=` tags — train split, 997
sentences:

| Tag | Tokens | % |
|---|---:|---:|
| `ar_dz` (Algerian Darija) | 9,408 | 65.0% |
| `fr` (French) | 4,769 | 33.0% |
| `ar_msa` (Modern Standard Arabic) | 528 | 3.6% |
| `en` | 83 | 0.6% |
| `es` | 25 | 0.2% |
| untagged (`_`) | 162 | 1.1% |
| `de`, `pt` | 7 | <0.1% |

(Percentages sum over ~14,459 non-multiword tokens, so >100% of the
"main" categories is just rounding — dev/test splits show the same
roughly 2:1 Darija:French ratio.) This is a genuine, annotated
code-switching signal — directly useful for validating this project's own
Darija/French code-switching assumptions (`Project_context.md`'s "heavily
code-switched with French" framing) against real linguistic annotation
rather than this project's own heuristics.

## Labels

**`Narabizi/sentiment/`** (joined to the treebank by `sent_id`, 998 train
/ 137 dev / 144 test = 1,279 labeled sentences — one more than the
treebank's 1,278 total, likely a duplicate `sent_id` in the label file;
not chased further):

| Label | Count | % |
|---|---:|---:|
| POS | 382 | 29.9% |
| NEG | 352 | 27.5% |
| MIX | 313 | 24.5% |
| NEU | 232 | 18.1% |

4-way, reasonably balanced, and notably includes **MIX** (mixed
sentiment) as its own class — most sentiment sets collapse to
POS/NEG/NEU only.

**`Narabizi/topic/`** (same join, same row counts):

| Label | Count | % |
|---|---:|---:|
| SPORT | 464 | 36.3% |
| NONE | 370 | 28.9% |
| SOCIETAL | 260 | 20.3% |
| POLITICS | 107 | 8.4% |
| PRAYER | 56 | 4.4% |
| RELIGION | 22 | 1.7% |

Sport-heavy — consistent with Twitter-scraped Algerian content generally
(football commentary is a dominant genre in this project's YouTube corpus
too, per `CLAUDE.md`'s channel categories).

**`sentiment/ar_dz`** (the *separate* football/political resource, binary,
731 rows total): 351 label-`0`, 380 label-`1` — roughly balanced, but the
CSVs don't state which integer means positive/negative; would need to
check the paper or spot-check a few rows against sentiment before using
this for anything beyond "there are two balanced classes."

## Auxiliary multilingual sentiment data (not Algerian — don't use as Darija data)

`sentiment/` also ships five other languages, sized for zero-shot
transfer experiments (train the sentiment model on a "helper" language
close in script or typology to Arabizi, test zero-shot on `ar_dz`/
`ar_na`/Narabizi) per the paper's own LaTeX table templates found in
`experiments/data_exploration/{pos,sentiment}_table.txt`:

| Language | Train | Dev | Test |
|---|---:|---:|---:|
| `ar` (MSA — book-review sentiment, e.g. reviews of the novel عزازيل) | 35,843 | 5,134 | 10,076 |
| `he` (Hebrew) | 8,702 | 1,240 | 2,492 |
| `ur` (Urdu) | 686 | 98 | 196 |
| `fa` (Persian) | 615 | 88 | 176 |
| `mt` (Maltese) | 504 | 72 | 145 |

These are large, well-formed, unrelated-domain datasets (`ar`'s book
reviews in particular are long-form, formal MSA — nothing like the
treebank's tweets) — useful only if this project ever wants to replicate
the paper's own cross-lingual transfer experiments, not as additional
Darija corpus material.

## Format notes / gotchas

- **CoNLL-U parsing needs care with multi-word tokens.** Rows like
  `5-6\tfiparout\t...` group two sub-tokens (`fi` + `parout`) under one
  surface form — they have no `lang=` tag of their own and must be
  skipped when counting tokens, or code-switching stats double-count
  (`explore_narabizi.py`'s `parse_conllu()` does this correctly — see
  the `"-" in cols[0]` check).
- **`Narabizi/pos/*.conllu` uses an extended, non-standard column
  layout** (extra French-translation and Arabic-transliteration columns
  beyond the standard 10-field CoNLL-U spec) — don't feed it directly to
  a standard CoNLL-U parser expecting exactly 10 tab-separated fields;
  use `ud/ar_dz/` or `ud/ar_na/` instead if standard-format CoNLL-U is
  needed, since those are confirmed identical in content.
- **Small scale overall**: the core treebank is 1,278 sentences — two
  orders of magnitude smaller than Twifil's 6,000 tweets, and far smaller
  than this project's own corpus (millions of docs). Its value here is
  annotation quality and genuine Algerian-Darija ground truth, not
  volume — e.g. a small labeled eval set, not additional pretraining
  data.

## Files

- `scripts/explore_narabizi.py` — parses everything above and prints/
  writes the numbers in this doc; rerun to refresh.
- `data/narabizi_report.json` (gitignored) — same numbers as JSON.
- `data/` (gitignored) — the original downloaded repo's data files
  (treebank, sentiment, topic, auxiliary multilingual sentiment sets).
- `experiments/` (gitignored via its own dedicated rule, not the data
  pattern below) — the original paper's own reproduction scripts/
  notebooks, kept as-is, not modified or relied upon by
  `scripts/explore_narabizi.py`.

Only this doc and `scripts/` are tracked in git. `data/` is ignored via
`Benchmarks/*/data/*` in the repo's `.gitignore` — same "data ignored,
scripts/docs tracked" convention as the rest of this repo.
`experiments/` gets its own dedicated
`Benchmarks/NArabizi-main/experiments/*` rule instead, since it's
third-party reproduction code bundled with the download, not this
project's own work — same spirit as the data rule, just scoped
specifically to this folder rather than generalized to every future
benchmark (no reason to assume the next one will have this same
bundled-experiments-folder shape).
