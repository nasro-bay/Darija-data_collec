# Algerian Darija Text Corpus — Project Context

## Goal

Build a large, high-quality **text-only** corpus of Algerian Darija (Algerian
Arabic dialect) for **general LLM pretraining**. Algerian Darija is a
low-resource language variety: it has no standard orthography, is written in
both Arabic script and Arabizi (Latin script with numerals substituting for
sounds like 3 = ع, 7 = ح, 9 = ق), and is heavily code-switched with French.
There is currently no large public pretraining-scale corpus for it, so this
project collects one from scratch, source by source.

## Scope decisions (locked in)

- **Modality**: text only (no audio/speech).
- **Use case**: general LLM pretraining (not classification/annotation). No
  labels needed — just clean, deduplicated, high-coverage raw text.
- **Script policy**: preserve natural variation. Do **not** force
  normalization between Arabic script and Arabizi, and do **not** strip
  French code-switching. Real Darija online is a mix of all of this, and the
  corpus should reflect the true distribution. Each document should instead
  be **tagged** with metadata (detected script, rough dialect confidence)
  so downstream users can filter if they want.
- **Region**: no single-region restriction for now — collect broadly across
  Algerian sources; region can be tracked as metadata when inferable
  (channel/account location, dialect markers) but is not a filter.
- **Scale target**: realistic milestones, not a single huge number.
  Track progress in stages: 1M → 10M → 50M+ tokens. Every source will be
  scraped incrementally with token counts logged after cleaning.

## Overall pipeline (applies to every source, including YouTube)

1. **Collect** raw data via official API where possible (respect ToS and
   rate limits).
2. **Language/dialect filter** — distinguish Darija from Modern Standard
   Arabic (MSA), French, Kabyle/Tamazight, and other Arabic dialects.
   Start with a heuristic pass (Darija-specific function words: e.g. "wach",
   "kayen", "bezzaf", "raki", "chkoun", "khoya", "wallah", "hna", "kima",
   etc., plus Arabizi numeral patterns) before considering a trained
   classifier.
3. **Clean**:
   - Deduplicate (exact + near-duplicate via minhash/simhash — social
     content has heavy repost/copy-paste volume)
   - Strip boilerplate/spam/bot patterns (engagement bait, crypto spam,
     "subscribe" comments)
   - Remove PII (phone numbers, emails, full names in personal contexts)
   - Minimum length filter (drop emoji-only / single-word noise)
   - Fix encoding issues (mojibake, inconsistent UTF-8)
4. **Anonymize**: replace user mentions, emails, and links with generic
   placeholders (do not delete — preserve the structural signal that a
   mention/link existed).
5. **Tag metadata per document**: source, source-type, scrape date,
   detected script (Arabic / Latin / mixed), rough dialect-confidence
   score, char/token count, dedup hash.
6. **Store** in a consistent format (JSONL preferred) with the schema
   below.
7. **Log** running token counts per source and cumulative total, so
   progress against the scale milestones is always visible.

## Data schema (JSONL, one object per document)

```json
{
  "id": "yt_<video_id>_<comment_id>",
  "text": "raw comment text, unmodified except anonymization",
  "source": "youtube",
  "source_type": "comment",
  "video_id": "...",
  "channel": "...",
  "scrape_date": "YYYY-MM-DD",
  "script": "arabic | latin | mixed",
  "darija_confidence": 0.0,
  "char_count": 0,
  "token_count": 0,
  "dedup_hash": "..."
}
```

## Current phase: YouTube comments collection

This is the **first source** to implement end-to-end (scrape → filter →
dedupe → clean → store) before parallelizing to other sources. It should
produce a working, reusable pipeline template.

### Why YouTube first
High yield of authentic Darija, especially in comment sections of Algerian
vloggers, comedy channels, football/sports commentary, and news channel
videos — comments skew far more dialectal than video descriptions or
formal captions.

### Requirements
- Use the **YouTube Data API v3** (free quota: 10,000 units/day; listing a
  comment thread page costs ~1 unit, so this comfortably supports
  thousands of comments/day within the free tier).
- Target **Algerian channels/videos**: vloggers, comedy/sketch channels,
  football commentary and reaction channels, news channel comment
  sections, podcast channels.
- Collect **top-level comments and replies** (replies often contain more
  casual/conversational Darija).
- Store raw scrape output separately from cleaned output, so cleaning
  logic can be iterated on without re-scraping.
- Log source video metadata (video ID, channel, title, scrape date) per
  batch for traceability and future removal requests.

### Deliverables for this phase
1. A YouTube scraper script (API-based, quota-aware, resumable).
2. A first raw batch pulled from a seed list of Algerian channels.
3. The language/dialect heuristic filter applied to that batch.
4. Deduplication + cleaning applied.
5. Output written in the JSONL schema above.
6. A short log/report: videos scraped, comments collected, comments
   retained after filtering, estimated token count, script distribution
   (Arabic vs Latin vs mixed).

## Legal / ethical notes

- Use the official YouTube Data API (not raw page scraping) to stay within
  ToS.
- Anonymize personal data before any release (mentions, emails, phone
  numbers, full names).
- Track per-source ToS compliance as more sources are added later
  (Reddit, forums, etc.).
- Decide on a release license once the corpus reaches a usable size
  (research-only vs. open); not required to finalize before this phase.

## Not in scope for this phase (future work, not now)

- Other sources (Reddit, forums, Instagram/TikTok, song lyrics, subtitles).
- A trained dialect classifier (heuristic filter is enough for now).
- Any annotation/labeling.
- Speech/audio collection.
