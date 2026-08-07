# Data Source: Djelfa.info Forum (منتديات الجلفة)

## Overview

This is the next data source for the Algerian Darija text corpus project
(see `PROJECT_CONTEXT.md` for full project scope/goal/schema — this file
adds a new source, same pipeline).

**URL**: https://www.djelfa.info/vb/
**Platform**: vBulletin (Powered by vBulletin, Copyright © 2018 vBulletin
Solutions, Inc. — legacy version, HTML structure should be stable/standard
vBulletin markup)
**Status**: Active, very high traffic.

## Why this source

Large, long-running, general-purpose Algerian forum. Scale observed at
time of writing:
- **1,687,367 topics**
- **21,655,008 posts**
- **669,762 registered members**
- **4,049 users online concurrently** at time of check (186 members,
  3,863 guests)

This is significantly larger than other Algerian forums evaluated so far
(algerie-dz.com, forum-algerie.com, algdz.com, forumdz.com). High post
volume + high concurrent activity means both large historical archives
and ongoing fresh content.

## Scope: collect from ALL sections, no filtering by topic at scrape time

Do not pre-filter or skip subforums based on assumed dialect density.
Scrape broadly across the whole forum structure. Dialect/language
filtering happens **downstream** in the existing pipeline (heuristic
Darija filter step), not by deciding in advance which subforums to
include. Every subforum is a candidate source — religious, academic,
technical, and administrative sections included. The goal is maximum
raw coverage; filtering is a separate, later stage.

## Forum structure (top-level categories observed)

The forum is organized into major categories, each with multiple
subforums. Known categories include (not exhaustive — the scraper should
discover the full subforum tree programmatically, this is just a map of
what exists):

- **خيمة الجلفة** (Djelfa Tent) — welcome/general chat, very high volume
  (33,017 topics / 683,799 posts)
- **الجلفة للنقاش الجاد** — serious discussion
- **منتديات الدين الإسلامي الحنيف** — Islamic religion (general, Quran/Sunnah,
  fiqh, Islamic history, etc.)
- **منتدى الحياة اليومية** — daily life (personal experiences, services,
  Q&A, housing, electronics repair, cars, admin documents, tourism)
- **منتديات الجزائر** — Algeria-focused (history, personalities, **اللهجة
  الجزائرية / Algerian Dialect subforum — 1,992 topics / 69,256 posts**,
  local culture/traditions, old photo archives, cities/regions)
- **منتديات الأنساب، القبائل و البطون** — tribes/lineage/ancestry
- **منتديات الأخبار... النُصرة و قضايا الأمّة** — news (Palestine support,
  Algeria news, political discussion, Arab/world news)
- **منتدى الأسرة و المجتمع** — family and society (society, "my problem"
  advice section, marital life)
- **منتدى شقائق الرجال** — women's forum (cooking, handicrafts, sewing,
  home, decor)
- **المنتدى الإسلامي للنّساء** — Islamic forum for women
- **منتديات انشغالات الأسرة التربوية** — education staff concerns
- **منتديات التعليم الإبتدائي / المتوسط / الثانوي** — primary/middle/
  secondary education
- **منتديات التكوين و التعليم عن بعد** — distance training/education
- **منتديات الجامعة و البحث العلمي** — university and scientific research
- **منتدى أساتذة التعليم العالي و البحث العلمي** — higher-ed faculty
- **منتدى التوظيف و المسابقات** — employment and competitions
- **Forum Français** — French-language subforum (Islam, presentations,
  culture debates, news, history, cuisine)
- **English Forum** — English-language subforum
- **منتديات الثقافة و الأدب** — culture and literature (general culture,
  Arabic language, literary "tent", creativity, reposted prose/poetry
  including dialectal poetry)
- **منتديات الثقافة الطبية و العلوم** — medical culture and science
- **منتدى المال و الأعمال** — money and business (commercial forum,
  banking, online shopping, earning sites)
- **منتديات التقنية** — tech (Linux, programming, security, software)
- **منتديات التصميم و الجرافيكس** — design/graphics
- **منتديات أصحاب المواقع** — website owners
- **منتديات التقنية الفضائية والستلايت** — satellite tech
- **منتدى عالم الرياضة** — sports (Algerian sports news, Arab/world sports,
  training)
- **منتديات عامة للترفيه و التسلية** — general entertainment (jokes/
  anecdotes — 27,458 topics / 488,736 posts, strange/funny photos,
  games/puzzles)
- **المنتدى الإداري** — administrative (forum admin, announcements, polls)

Some subforums are marked **خاص** (private/restricted access) — these
require membership/permissions and should be skipped unless we have an
account with appropriate access.

## Scraping requirements

1. **Respect `robots.txt`**: check `https://www.djelfa.info/robots.txt`
   before building the scraper and honor any disallowed paths.
3. **No login required** for reading public (non-خاص) sections — scraper
   should work anonymously for those; skip خاص sections entirely (no
   attempt to bypass access restrictions).
4. **Crawl approach**:
   - Discover the full subforum tree from the forum index page
   - For each subforum, paginate through thread listings
   - For each thread, paginate through posts
   - Standard vBulletin pagination/URL patterns apply
5. **Extract per post**:
   - Post text (raw, unmodified except later anonymization step)
   - Thread title
   - Subforum name/category (for metadata — useful for later analysis
     even though we're not filtering by it at scrape time)
   - Author (to be anonymized downstream, not stored as identifying info
     in the final cleaned dataset)
   - Post timestamp
   - Thread/post URL (for traceability, per project logging requirements)

## Output

Raw scraped output goes into the same pipeline as other sources:
collect → language/dialect heuristic filter → dedupe → clean →
anonymize → tag metadata → store in the project's JSONL schema
(see `PROJECT_CONTEXT.md`), with `"source": "djelfa_info"` and
`"source_type": "forum_post"`.

## Deliverables for this phase

1. A forum scraper script (respecting robots.txt and rate limits,
   resumable/checkpointable given the scale of this forum).
2. Full subforum tree discovery (programmatic, not hardcoded from the
   list above — that list is a reference snapshot, not a fixed target
   set).
3. A first raw batch pulled from a representative sample of subforums
   across categories.
4. Existing heuristic Darija filter applied to the batch.
5. Dedup + cleaning applied.
6. Output in the JSONL schema.
7. A short log/report: subforums crawled, threads/posts collected,
   posts retained after filtering, estimated token count, and (since we
   are not pre-filtering by section) a breakdown of retained-post count
   by source subforum/category — useful to see empirically which
   sections turned out most Darija-dense, after the fact rather than by
   assumption.