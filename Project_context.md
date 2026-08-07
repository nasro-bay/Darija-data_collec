# Algerian Darija Text Corpus — Project Context

## Goal

Build a large, high-quality **text-only** corpus of Algerian Darija
(Algerian Arabic dialect) for **general LLM pretraining**. Algerian Darija
is a low-resource language variety: it has no standard orthography, is
written in both Arabic script and Arabizi (Latin script with numerals
substituting for sounds like 3 = ع, 7 = ح, 9 = ق), and is heavily
code-switched with French. There is currently no large public
pretraining-scale corpus for it, so this project collects one from
scratch, source by source.

## Scope decisions (locked in)

- **Modality**: text only (no audio/speech).
- **Use case**: general LLM pretraining (not classification/annotation).
  No labels needed — just clean, deduplicated, high-coverage raw text.
- **Script policy**: preserve natural variation. Do **not** force
  normalization between Arabic script and Arabizi, and do **not** strip
  French code-switching. Real Darija online is a mix of all of this, and
  the corpus should reflect the true distribution. Each document should
  instead be **tagged** with metadata (detected script, rough dialect
  confidence) so downstream users/models can filter if they want.
- **Region**: no single-region restriction — collect broadly across
  Algerian sources; region can be tracked as metadata when inferable, but
  is not a filter.
- **Topic/section filtering**: don't pre-filter by subject matter at
  collection time (e.g. within a forum, don't skip whole sections assumed
  to be "less dialectal"). Collect broadly; let the language/dialect
  filter (step 2 of the pipeline below) do the actual sorting downstream,
  after collection. This applies across all sources, not just forums.
- **Scale target**: realistic milestones, not one big number. Track
  progress in stages: 1M → 10M → 50M+ tokens. Every source is scraped
  incrementally with token counts logged after cleaning.

## Overall pipeline (applies to every source)

1. **Collect** raw data via official API where one exists; otherwise via
   direct scraping that respects `robots.txt` and rate limits.
2. **Language/dialect filter** — distinguish Darija from Modern Standard
   Arabic (MSA), French, Kabyle/Tamazight, and other Arabic dialects.
   Start with a heuristic pass (Darija-specific function words: e.g.
   "wach", "kayen", "bezzaf", "raki", "chkoun", "khoya", "wallah", "hna",
   "kima", etc., plus Arabizi numeral patterns) before considering a
   trained classifier.
3. **Clean**:
   - Deduplicate (exact + near-duplicate via minhash/simhash — social
     content has heavy repost/copy-paste volume)
   - Strip boilerplate/spam/bot patterns (engagement bait, crypto spam,
     "subscribe"/signature spam)
   - Remove PII (phone numbers, emails, full names in personal contexts)
   - Minimum length filter (drop emoji-only / single-word noise)
   - Fix encoding issues (mojibake, inconsistent UTF-8)
4. **Anonymize**: replace user mentions, emails, and links with generic
   placeholders (do not delete — preserve the structural signal that a
   mention/link existed).
5. **Tag metadata per document**: source, source-type, scrape date,
   detected script (Arabic / Latin / mixed), rough dialect-confidence
   score, char/token count, dedup hash, plus source-specific fields
   (e.g. subforum/category for forums, channel for YouTube).
6. **Store** in a consistent format (JSONL) with the schema below.
7. **Log** running token counts per source and cumulative total, so
   progress against the scale milestones is always visible. For sources
   with internal sections/categories (e.g. forum subforums), also log a
   post-filtering breakdown by section — this tells us empirically which
   sections were Darija-dense, rather than assuming it upfront.

## Data schema (JSONL, one object per document)

```json
{
  "id": "source-prefixed unique id, e.g. yt_<video_id>_<comment_id> or djelfa_<post_id>",
  "text": "raw text, unmodified except anonymization",
  "source": "youtube | reddit | djelfa_info | ...",
  "source_type": "comment | forum_post | ...",
  "source_metadata": {
    "video_id": "... (youtube)",
    "channel": "... (youtube)",
    "subforum": "... (forums)",
    "thread_url": "... (forums)"
  },
  "scrape_date": "YYYY-MM-DD",
  "script": "arabic | latin | mixed",
  "darija_confidence": 0.0,
  "char_count": 0,
  "token_count": 0,
  "dedup_hash": "..."
}
```

## Sources — status overview

| Source | Status | Access method | Notes |
|---|---|---|---|
| YouTube comments | **Active — current build phase** | YouTube Data API v3 (free quota) | See "YouTube" section below |
| Reddit | **Blocked / deprioritized** | Official API, self-service closed | See "Reddit" section below — do not build against this yet |
| djelfa.info forum | **Active — next phase** | Direct scraping (no official API) | See "Forums" section below; largest forum source identified so far |
| Other Algerian forums (algerie-dz.com, forum-algerie.com, algdz.com, forumdz.com) | Candidate / secondary | Direct scraping | Smaller than djelfa.info; not yet started, revisit after djelfa.info pipeline is validated |

---

## Source: YouTube comments

**Why**: high yield of authentic Darija, especially in comment sections
of Algerian vloggers, comedy channels, football/sports commentary, and
news channel videos — comments skew far more dialectal than video
descriptions or formal captions.

**Access**: YouTube Data API v3. Free quota: 10,000 units/day; listing a
comment thread page costs ~1 unit, so this comfortably supports
thousands of comments/day within the free tier.

**Targets**: Algerian channels/videos — vloggers, comedy/sketch channels,
football commentary and reaction channels, news channel comment
sections, podcast channels.

**Collection notes**:
- Collect both top-level comments and replies (replies often contain
  more casual/conversational Darija).
- Store raw scrape output separately from cleaned output, so cleaning
  logic can be iterated on without re-scraping.
- Log source video metadata (video ID, channel, title, scrape date) per
  batch for traceability and future removal requests.

**Deliverables for this phase**:
1. YouTube scraper script (API-based, quota-aware, resumable).
2. First raw batch pulled from a seed list of Algerian channels.
3. Language/dialect heuristic filter applied to that batch.
4. Deduplication + cleaning applied.
5. Output written in the JSONL schema above.
6. Short log/report: videos scraped, comments collected, comments
   retained after filtering, estimated token count, script distribution
   (Arabic vs Latin vs mixed).

---

## Source: Reddit — BLOCKED, do not build against this yet

**Status**: Do not implement a Reddit scraper/PRAW pipeline at this time.

**Why blocked**:
1. Self-service Reddit API app registration is currently closed. New
   OAuth access requires manual approval under Reddit's "Responsible
   Builder Policy" (introduced ~March 2026), with no guaranteed timeline.
2. More importantly: Reddit's Responsible Builder Policy explicitly
   states that using Reddit data to **train machine learning or AI
   models** — commercial or non-commercial — requires **express written
   approval**, separate from general API access. This project's stated
   purpose (corpus for LLM pretraining) falls directly under that
   restriction.
3. The policy's Researcher provisions (Reddit for Researchers Program)
   also require not retaining data beyond the immediate project and
   re-running queries against the latest export to reflect deletions —
   which conflicts with building a persistent, reusable, potentially
   released corpus.

**If revisited later**: would require submitting an explicit written
request describing the AI-training use case and getting approval before
any collection — not just standard API app registration. Track this as a
separate legal/compliance task, not a scraping task.

---

## Source: djelfa.info forum (منتديات الجلفة) — next phase

**URL**: https://www.djelfa.info/vb/
**Platform**: vBulletin (standard/legacy markup).
**Status**: Active, very high traffic — largest Algerian forum
identified so far.

**Scale** (observed at time of writing):
- 1,687,367 topics
- 21,655,008 posts
- 669,762 registered members
- Thousands of concurrent users typically online

**Scope**: Collect broadly across **all** subforums — do not pre-filter
by assumed dialect density. This includes religious, academic, technical,
and administrative sections, not just casual/social ones. Filtering
happens downstream in the pipeline's language/dialect step, not by
excluding sections at scrape time.

**Notable subforum**: منتدى اللهجة الجزائرية (Algerian Dialect forum,
~1,992 topics / ~69,256 posts) — a subforum specifically dedicated to
dialect discussion and regional variation (Dziri/Algiers, Kabyle-
influenced, Chaoui, Mzabi, Nayli, Eastern and Western dialects). High
expected Darija density, but treat as one of many sections, not the only
target.

**Known forum structure** (reference snapshot — scraper should discover
the actual subforum tree programmatically, not rely on this list):
Djelfa Tent / general chat, serious discussion, Islamic religion
(general, Quran/Sunnah, fiqh, history), daily life (personal experiences,
services, Q&A, housing, electronics, cars, admin documents, tourism),
Algeria-focused (history, dialect, culture/traditions, old photos,
cities/regions), tribes/lineage, news (Palestine, Algeria, politics,
world), family and society, women's forum (cooking, handicrafts, sewing,
home, decor), Islamic forum for women, education staff concerns, primary/
middle/secondary education, distance training, university and scientific
research, higher-ed faculty, employment, French-language subforum,
English-language subforum, culture and literature (incl. dialectal
poetry/prose), medical culture and science, money and business, tech
(Linux, programming, security, software), design/graphics, website
owners, satellite tech, sports, general entertainment (jokes, funny
photos, games), administrative/meta.

Some subforums are marked **خاص** (private/restricted) — skip these
entirely; no attempt to bypass access restrictions.

**Scraping requirements**:
1. Check and respect `https://www.djelfa.info/robots.txt`.
2. Rate-limit requests — this is a live community forum, not an API.
3. No login needed for public (non-خاص) sections; scraper works
   anonymously for those.
4. Crawl approach: discover full subforum tree from the forum index →
   paginate thread listings per subforum → paginate posts per thread.
5. Extract per post: text, thread title, subforum/category (metadata),
   author (anonymized downstream), timestamp, thread/post URL (for
   traceability).

**Deliverables for this phase**:
1. Forum scraper script (robots.txt-compliant, rate-limited, resumable/
   checkpointable given the forum's scale).
2. Programmatic subforum tree discovery.
3. First raw batch from a representative sample of subforums across
   categories.
4. Heuristic Darija filter applied.
5. Dedup + cleaning applied.
6. Output in the JSONL schema, `"source": "djelfa_info"`,
   `"source_type": "forum_post"`.
7. Log/report: subforums crawled, threads/posts collected, posts
   retained after filtering, estimated token count, and a breakdown of
   retained-post count by subforum/category (empirical, not assumed).

---

## Other candidate forums (secondary, not yet started)

Identified but not yet scraped — revisit once the djelfa.info pipeline
is validated and can be reused/adapted:
- **algerie-dz.com/forums** — active, vBulletin, general topics; confirmed
  genuine Darija/Arabizi content in casual threads.
- **forum-algerie.com** — active, general discussion, sports,
  entertainment, science.
- **algdz.com/forums** (ALG DZ) — active, travel/culture/food/sports/tech.
- **forumdz.com** — active but tech/telecom-focused; likely lower dialect
  density, useful mainly for diversity/volume, lower priority.

---

## Legal / ethical notes (project-wide)

- Use official APIs where available (YouTube) and respect their terms.
- For direct scraping (forums), respect `robots.txt` and rate limits;
  no bypassing of access controls or private/restricted sections.
- Anonymize personal data before any release (mentions, emails, phone
  numbers, full names) — applies to every source.
- Track per-source ToS/legal status explicitly (see status table above);
  do not build collection scripts for sources marked blocked without
  resolving the underlying legal question first.
- Release license (research-only vs. open) not yet finalized — decide
  once the corpus reaches a usable size; not required to finalize before
  active collection phases.

## Not yet in scope (future work)

- Instagram/TikTok, song lyrics, subtitles as additional sources.
- A trained dialect classifier (heuristic filter is enough for now).
- Any annotation/labeling.
- Speech/audio collection.
- Actual model pretraining pipeline/architecture (separate track — see
  hardware note below; not part of the data collection deliverables).

## Hardware note (for later phases, not data collection)

Target training hardware for eventual pretraining: RTX 2060 (6GB VRAM,
Turing architecture). This favors a smaller-than-BERT-Base architecture
(e.g. ~6 layers, 512 hidden dim, ~16-24K vocab), short max sequence
length (64-96 tokens, matched to short social/forum text), fp16 mixed
precision, and gradient accumulation. Not required knowledge for the
data collection phase, but corpus stats (typical document length, token
distribution) gathered during cleaning should be reported since they'll
inform these choices later.