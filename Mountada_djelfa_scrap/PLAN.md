# djelfa.info Forum Scraper — Implementation Plan

## Context

`Readme.md` (this folder) + `Project_context.md` (project root) define this
source: djelfa.info, a huge vBulletin forum (1.68M topics / 21.6M posts),
next source for the Darija corpus after YouTube. Collection must be
**broad, unfiltered by section** — dialect filtering happens downstream,
not by excluding subforums up front.

Two access findings from checking `https://www.djelfa.info/robots.txt` and
a live request before writing any code (both required regardless of
ownership — same due-diligence step this project already applies to every
source):

- **robots.txt** explicitly disallows `ClaudeBot` site-wide and sets
  `Content-Signal: ai-train=no, use=reference` for everyone — normally a
  hard blocker (same treatment as the Reddit entry in
  `Project_context.md`). **You've confirmed you own/operate djelfa.info**,
  which is what changes this: the restriction is the site owner reserving
  rights against *other* parties' AI-training use, and doesn't apply to
  the owner's own use of their own data. Recorded here for traceability,
  same as every other source's legal status is tracked.
- **Cloudflare is actively challenge-gating the site** (`Cf-Mitigated:
  challenge`, the "Just a moment..." interstitial) — confirmed with both a
  plain and a browser-spoofed `curl` request; both got HTTP 403. This
  blocks any plain HTTP scraper (`requests`/`httpx`/`curl`) outright,
  independent of the robots.txt question. This is a technical constraint
  the plan has to design around (see below) — it isn't optional.

**Design response to the Cloudflare gate**: solve the JS challenge once
per session with a real (headless) browser, harvest the resulting
`cf_clearance` cookie + matching User-Agent, then do the actual bulk
crawling cheaply with a plain HTTP client reusing that cookie — refreshing
via the browser only when the cookie expires or a challenge reappears.
Running a full browser for every one of millions of page fetches isn't
viable; solving the challenge once per session and reusing the session is.
If Cloudflare ever escalates to an interactive CAPTCHA (Turnstile) instead
of auto-clearing, that's a signal to add a Cloudflare-side allowlist rule
for the scraper's IP (you control this, since you operate the site) rather
than attempting to automate past a CAPTCHA — not part of this plan.

**Structure**: mirrors the already-built `Youtube_scrap/` layout (its own
`src/`, `scripts/`, `config/`, `data/`, `tests/`) rather than sharing code
directly — same pattern the project has already settled into per-source.

**Schema note**: `Project_context.md`'s current schema nests source-specific
fields under `source_metadata` (`{"subforum": ..., "thread_url": ...}`),
which differs from `Youtube_scrap`'s flat schema (`video_id`/`channel` as
top-level fields, written before this nested convention existed). This
plan builds djelfa's `schema.py` against the current nested spec, since
that's the authoritative version now. Reconciling `Youtube_scrap`'s schema
to match is a separate, later cleanup — out of scope here.

## Shared foundations (used by both sub-plans)

```
Mountada_djelfa_scrap/
  requirements.txt              # requests, beautifulsoup4, lxml, playwright, pyyaml, tqdm
  src/darija_forum/
    __init__.py
    session.py       # Playwright challenge-solver: harvests cf_clearance cookie + UA
    http_client.py    # requests.Session wrapper using the harvested session; detects
                       # a renewed challenge and signals "needs refresh" rather than
                       # silently parsing challenge HTML as content
    state.py          # persisted, resumable state (atomic writes — same pattern as
                       # Youtube_scrap/src/darija_corpus/_atomic.py, reused as-is)
    parse.py           # HTML parsing: thread-listing rows, post extraction
                        # (selectors TBD — see "First implementation step" below)
    dedup.py            # near-dup MinHash/LSH — same approach as Youtube_scrap's,
                         # ported essentially unchanged
    schema.py            # builds the nested-schema document per Project_context.md
    pipeline.py            # dedup -> JSONL -> global log, + per-subforum breakdown
  data/
    raw/djelfa/<subforum_id>/<thread_id>.jsonl
    state/forum_tree.json        # discovery output (sub-plan 1)
    state/scrape_targets.json    # flattened non-private forum_id list (sub-plan 1 -> 2)
    state/crawl_state.json       # subforum/thread pagination progress (sub-plan 2)
    state/session.json           # harvested cf_clearance cookie + UA + solved-at time
    processed/batch_<date>.jsonl
    logs/log.json
```

**Rate limiting**: no `Crawl-delay` in robots.txt, but at this scale
(21M+ posts) a fixed, conservative delay between requests (configurable,
default ~1–2s, sequential — no concurrency in this first build) is
non-negotiable regardless of ownership, so the crawl doesn't degrade the
live site for real users.

**First implementation step, before writing selectors**: manually solve
the Cloudflare challenge once (e.g. open the site in a real browser) and
inspect the actual rendered markup for a subforum listing and a thread
page. `parse.py`'s selectors below are written against the *standard*
vBulletin URL scheme (`forumdisplay.php?f=<id>&page=<n>`,
`showthread.php?t=<id>&page=<n>`) as a starting assumption, not verified
markup — confirm both the URL scheme and the actual post/thread-row HTML
structure against the real site before relying on them, since "legacy,
should be standard" in the Readme is a guess, not a guarantee.

---

# Sub-plan 1: Discovery

## Goal

Programmatically map the full subforum tree (category → subforum →
nested child-forums), flagging private (`خاص`) sections to exclude —
producing the input list for sub-plan 2. This is a small, one-time
(or occasionally-rerun) crawl — dozens to ~100 pages, not the bulk of the
work.

## Steps

1. **`session.py`**: Playwright launches headless Chromium, navigates to
   the forum index, waits for the page title to move off "Just a
   moment...", harvests cookies + the browser's own User-Agent string,
   writes them to `data/state/session.json` with a solved-at timestamp.
2. **`http_client.py`**: wraps `requests.Session`, preloaded with the
   harvested cookie + matching UA. Every response is checked for
   challenge markers (title/response signature) — if a challenge is
   detected mid-crawl (cookie expired), the client raises a
   `SessionExpiredError` rather than silently treating the interstitial
   HTML as real content; the caller re-runs `session.py` and retries.
3. **`discover.py`**:
   - Fetch the forum index; parse top-level category blocks and their
     subforum links.
   - For each subforum, follow its listing page; recursively follow any
     nested child-forum links found there (vBulletin sites sometimes
     nest sub-subforums a level or two deep).
   - Detect `خاص`/private sections (title marker, restricted-access CSS
     class, or a permission-denied response on access attempt) and flag
     `is_private: true` — never attempt to access their contents, no
     login flow is built.
   - Record, per subforum: `forum_id`, `title`, `category`, `url`,
     `is_private`, and visible thread/post counts if shown on the index
     (useful for later prioritization, not required for correctness).
4. **Output**:
   - `data/state/forum_tree.json` — full nested tree, for reference/debugging.
   - `data/state/scrape_targets.json` — flattened list of every
     non-private `forum_id`. Per the Readme's explicit "no filtering by
     topic," this is *all* discovered non-private subforums by default —
     unlike YouTube's channel-curation step, there's no human review gate
     here; that's a deliberate difference, not an oversight.
5. **`scripts/discover_forum_tree.py`**: CLI entrypoint. Prints a summary
   (categories found, subforums found, how many flagged private) and
   writes the two output files above.

## Verification

- `python scripts/discover_forum_tree.py --help` runs without errors.
- Live run against the real site: confirm the printed subforum count is
  in the right ballpark versus the Readme's reference snapshot (~25
  top-level categories), and spot-check a handful of `forum_id`s against
  known subforums (e.g. the "اللهجة الجزائرية" one named in the Readme).
- Confirm at least one `خاص` section gets correctly flagged private and
  excluded from `scrape_targets.json`.

---

# Sub-plan 2: Scraping

## Goal

For every subforum in `scrape_targets.json`, page through its thread
listing, and for every thread page through its posts, extracting: post
text, thread title, subforum/category, author (raw — anonymized in a
later pipeline stage, not at scrape time), timestamp, and thread/post URL
for traceability. Resumable, rate-limited, boundable per run (so the
"first representative batch across categories" deliverable doesn't
require attempting all 21M posts in one go).

## Steps

1. **`state.py`** (`data/state/crawl_state.json`):
   ```
   subforums[forum_id]: {next_thread_page, threads_found, capped_at, completed}
   threads[thread_id]:  {subforum_id, next_post_page, status: pending|done|error}
   pipeline: {processed_raw_files: []}
   ```
   Same atomic-write pattern as `Youtube_scrap`, and the same
   `capped_at`/`completed` distinction I just added there for channels —
   directly reused here for subforums, since it solves the identical
   problem: stopping at a cap now shouldn't block resuming further later
   with a higher (or no) cap.

2. **`parse.py`**:
   - `list_threads(forum_id, page) -> (thread_refs, has_next_page)` —
     parses a subforum listing page (`forumdisplay.php?f=<id>&page=<n>`,
     pending verification) into `{thread_id, title, url}` entries.
   - `list_posts(thread_id, page) -> (post_records, has_next_page)` —
     parses a thread page (`showthread.php?t=<id>&page=<n>`, pending
     verification) into `{post_id, author, timestamp, text, post_url}`
     entries.

3. **`scrape.py`**:
   - `scrape_thread(client, state, raw_dir, thread_id, subforum_id)` —
     pages through one thread's posts, appending each page's records
     immediately to `data/raw/djelfa/<subforum_id>/<thread_id>.jsonl`
     (crash-safe, same append-per-page pattern as the YouTube scraper).
     Marks the thread `done`/`error` in state.
   - `scrape_subforum(client, state, raw_dir, forum_id, *, max_threads=None)`
     — pages through the thread listing, calling `scrape_thread` for
     each new thread_id; enforces `max_threads` with the same
     cap-now-resume-later semantics as the YouTube channel walker
     (a `SessionExpiredError` from the HTTP client propagates up exactly
     like `QuotaExceededError` did for YouTube — state is already saved
     incrementally, so a rerun after refreshing the session resumes
     cleanly).

4. **`schema.py`**: builds the nested schema from `Project_context.md`:
   `id` (`djelfa_<post_id>`), `text`, `source: "djelfa_info"`,
   `source_type: "forum_post"`,
   `source_metadata: {subforum, thread_title, thread_url, post_url, author}`,
   `scrape_date`, `script`/`darija_confidence` (`null` — deferred, same as
   `Youtube_scrap`), `char_count`, `token_count`, `dedup_hash`.

5. **`pipeline.py`**: near-dup MinHash/LSH dedup (same design as
   `Youtube_scrap/src/darija_corpus/dedup.py`, ported essentially
   unchanged — forum reposts/copy-paste are exactly the kind of thing it
   already handles well) → `data/processed/batch_<date>.jsonl` → appends
   to `data/logs/log.json`. **Additionally** (explicit deliverable for
   this source): a per-subforum retained-post breakdown in the log, so
   which sections are actually Darija-dense is observed empirically
   rather than assumed up front.

6. **`scripts/run_scrape.py`**: CLI. Reads `scrape_targets.json`, walks
   subforums in order, skips anything flagged private, supports
   `--max-threads-per-subforum` and a global `--max-pages` run budget
   (mirrors `--max-videos` from the YouTube channel scraper) for
   incremental, boundable runs. Resumable exactly like the YouTube
   scripts: interrupt anytime, rerun the same command to continue.

7. **`scripts/run_pipeline.py`**: mirrors `Youtube_scrap`'s, adapted for
   the nested schema and the per-subforum breakdown report.

## Deferred (not in this pass)

- No login flow / no access to `خاص` sections.
- No concurrency — sequential requests only, for this first build.
- No headless-browser fallback for an escalated interactive CAPTCHA — if
  that happens, it means a Cloudflare-side allowlist rule is needed
  instead (owner-side fix, not a scraper feature).
- Darija heuristic filter / cleaning stage — same as `Youtube_scrap`,
  deferred to a later phase, reused once built rather than duplicated.

## Verification

- Unit tests for `parse.py`'s `list_threads`/`list_posts` against saved
  HTML fixtures (captured after the challenge is solved once) — offline,
  no live requests needed per test run, same approach as the mocked
  `YouTubeClient` tests already written for `Youtube_scrap`.
- Live check: a scraping run capped to one subforum and a handful of
  threads (`--max-threads-per-subforum 3`), confirming raw JSONL, a
  successful pipeline run, and valid nested-schema output end to end —
  before scaling up to a real "representative batch across categories."
