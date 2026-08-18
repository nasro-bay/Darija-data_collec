# Twifil — benchmark notes

First entry in `Benchmarks/`. Loaded via
`datasets.load_dataset("arbml/Twifil")` — see
`scripts/explore_twifil.py` for the exploration script this file's
numbers come from (rerun it to refresh them; nothing here is hand-typed
from memory).

## What it is

~6,000 tweets (single `train` split, no other splits) scraped in January
2018, with sentiment/emotion annotations. HF's own dataset card is mostly
`[More Information Needed]` placeholders — this doc is the result of
inspecting the actual data directly, not trusting the card.

**Confirmed via HF card + this project's own inspection**: North African
Arabic dialects broadly — **Algerian, Moroccan, and Tunisian mixed
together**, not exclusively Algerian Darija, plus heavy French/English
code-switching (expected for Maghrebi Twitter). **Don't treat this as a
pure-Darija benchmark without filtering** — see "Dialect/language mix"
below for how to approximate that filter.

Columns: `ID`, `Code`, `Post`, `lang`, `Created At`, `Followers Count`,
`Profile Link Color`, `Geo Enabled`, `Screen Name`, `Name`, `Profile Lang`,
`Polarity`, `Polarity Class`, `User Age`, `Emotion`, `Platform`.

## Task / labels

Primarily a **sentiment analysis** set:
- `Polarity` — a float sentiment score, **stored as a French-locale
  comma-decimal string** (e.g. `"3,43"`, `"-2,57"`) — `float(x)` will
  raise; use `float(x.replace(",", "."))`.
- `Polarity Class` — the categorical label: **Positive** (2,864),
  **Negative** (1,773), **Neutral** (1,363).

`Emotion` (18 distinct categories per the HF card) is present but **almost
useless as-is**: 5,944/6,000 rows (99.1%) are the literal string `"nan"`,
not a real null — only 56 rows have an actual emotion label, and even
those are messy (French words, trailing commas, e.g. `"colère, "`, and at
least one corrupted row repeating the same emotion 7 times). Not
recommended as a training/eval signal without a lot of cleanup first, and
even then the surviving sample size (56) is too small to be useful alone.

`Platform` is a constant (`"Twitter"` for all 6,000 rows) — not a useful
feature, all the metadata (`Followers Count`, `Screen Name`, `Geo
Enabled`, etc.) is Twitter-only.

## Data quality issues found

- **`Code` (tweet ID) has lost precision** — stored as a float64 in
  scientific notation (e.g. `"9.5283479986508e+17"`). Real Twitter
  snowflake IDs are 18-19 digit integers; float64 only reliably holds
  ~15-17 significant digits, so **the original tweet ID cannot be
  recovered exactly from this field**. Use `ID` (a clean sequential
  `1`-`6000` string) as the row identifier instead — it's what's actually
  unique and stable (verified: 6,000/6,000 unique).
- **794 rows (13.2%) are empty once `@mentions`/URLs are stripped** — pure
  reply-tag or pure link-share tweets with no other content (e.g. a post
  that's literally `"@user1 @user2 @user3"`). Same "near-empty, drop it"
  category this project's own `clean_text.py` filters out of the YouTube/
  djelfa corpora — Twifil has no such filtering applied.
- **65 rows are exact duplicates** (37 rows beyond first occurrence) — a
  few look like bot/spam activity (`"@BTS_twt I love you so much, please
  come to Algeria"` × 4), others are near-boilerplate replies (`"تم"` to
  different `@mentions`).
- **0 rows have an empty raw `Post`** — that part's clean.

## Script distribution — raw vs. cleaned differ enormously

This is the biggest thing to know before using this dataset. A naive
script check on the **raw** `Post` field says the dataset is mostly
*not* Arabic-script at all:

| Script (raw `Post`) | Count | % |
|---|---:|---:|
| Latin | 3,427 | 57.1% |
| Mixed | 2,300 | 38.3% |
| Arabic | 257 | 4.3% |
| Other/empty | 16 | 0.3% |

That's misleading — `@mentions` and `https://t.co/...` URLs are Latin-alphabet
by construction, so an otherwise-pure-Arabic tweet with a couple of
mentions gets bucketed as "mixed", and this dataset's tweets are reply-heavy
(lots of `@mentions`). **Stripping mentions/URLs first** (same idea as this
project's own `clean_text.py` placeholder approach) gives a very
different, much more trustworthy picture:

| Script (mentions/URLs stripped) | Count | % |
|---|---:|---:|
| Arabic | 2,375 | 39.6% |
| Latin | 2,378 | 39.6% |
| Other/empty | 1,065 | 17.8% |
| Mixed | 182 | 3.0% |

("Other/empty" here mostly *is* the 794 mention/URL-only rows above, plus
a few emoji-only posts.) So the dataset is roughly an even Arabic/Latin
split once you account for the noise, not the 57%-Latin picture the raw
text suggests.

## `lang` field reliability

The dataset's own `lang` column (auto-detected, presumably by Twitter's
own langid at scrape time) turns out to be a decent proxy for actual
script **once mentions/URLs are accounted for**: of the 2,462 rows tagged
`lang == "ar"`, 2,346 (95.3%) are genuinely Arabic-script once cleaned
(the raw-text check alone would've suggested only ~10% were, another
symptom of the mention/URL contamination above). `lang == "und"`
(1,244 rows, undetermined) is mostly the near-empty mention/URL-only
tweets (1,009/1,244 — makes sense, langid can't classify near-nothing).

| `lang` | n | Notes |
|---|---:|---|
| `ar` | 2,462 | 95.3% genuinely Arabic-script after cleaning |
| `und` | 1,244 | mostly near-empty (mention/URL-only) tweets |
| `fr` | 1,110 | |
| `en` | 733 | |
| (26 others) | 451 | Spanish, Indonesian, Turkish, Haitian Creole, Portuguese, Tagalog, etc. — noise from langid on short/ambiguous text |

## Dialect/language mix (approximate Darija filter)

There's no explicit dialect/country column, so there's no clean way to
isolate "just Algerian Darija" rows. Spot-checking a sample of `lang ==
"ar"` posts directly shows a real mix within that subset: some are Modern
Standard Arabic (formal religious quotations, political commentary), some
are genuinely Darija (`"راح دوكا الناس..."` — "دوكا" is a distinctly
Algerian word for "now"; a reference to someone from Sétif, Algeria in
another sample row). If you need this project's actual `script`/
`darija_confidence`-style filtering (see `Project_context.md`'s pipeline
step 2), that heuristic still isn't built anywhere in this repo yet —
using Twifil as a Darija-specific eval set means either building that
filter first, or accepting it as a noisier "North African Arabic dialect"
benchmark rather than a Darija-pure one.

## Files

- `scripts/explore_twifil.py` — downloads the dataset and computes every
  number in this doc; rerun to refresh.
- `data/twifil_report.json` (gitignored) — the same numbers as this doc,
  as JSON, for reuse without re-downloading.
- `data/twifil_cleaned.jsonl` (gitignored) — one row per tweet: `id`,
  `text` (URLs/mentions replaced with `[URL]`/`[MENTION]` placeholders,
  same convention as `Youtube_scrap`/`Mountada_djelfa_scrap`'s
  `clean_text.py` — **not** full cleaning, no dedup/elongation-collapsing),
  `lang`, `polarity_class`, `emotion` (`null` where the source was the
  literal `"nan"` string). Meant as a lighter-weight file for downstream
  scripts to load instead of re-running `load_dataset` every time.

Data files are gitignored (`Benchmarks/*/data/*` in the repo's
`.gitignore`) — only this doc and the scripts are tracked, same "data
ignored, scripts/docs tracked" convention as the rest of this repo.
