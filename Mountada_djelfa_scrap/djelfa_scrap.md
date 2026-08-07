# djelfa.info Forum Scraping — Execution Guide

How to run the pipeline in `src/darija_forum/` end to end: bootstrap a
session → discover subforums (one-time) → scrape threads/posts → dedupe
→ check output. See `PLAN.md` for the architecture and
`Project_context.md` (project root) for the overall corpus project.

## 0. Prerequisites

1. **Python deps**: `pip install -r requirements.txt`
2. **Session**: djelfa.info sits behind a Cloudflare JS challenge that
   blocks plain HTTP *and* Playwright automation outright (confirmed —
   see `PLAN.md`). The only working path is a session harvested from a
   real browser:
   1. Open `https://www.djelfa.info/vb/` in your normal browser, let it load.
   2. DevTools (F12) → **Network** tab → reload the page.
   3. Click the main document request (Type: document, near the top) →
      right-click → **Copy** → **Copy as cURL (bash)**.
   4. Give the whole curl command to `bootstrap_session.py` — it extracts
      the cookie header and User-Agent itself (ignoring everything else in
      there: `accept`, `referer`, `sec-ch-ua-*`, `--data-raw`, `origin`,
      ... — those belong to that one specific request, not the reusable
      session). Three ways to pass it, in order of preference:
      - **From a file** (most robust — no shell-quoting to worry about for
        a huge multi-line, quote-heavy paste): save the copied text to a
        file, then:
        ```
        python scripts/bootstrap_session.py --curl-file path/to/pasted_curl.txt
        ```
      - **Piped via stdin**:
        ```
        python scripts/bootstrap_session.py < path/to/pasted_curl.txt
        ```
      - **As a single argument** (fine for a quick paste, fiddly for a long one):
        ```
        python scripts/bootstrap_session.py --curl "curl 'https://...' -H '...' -b '...' ..."
        ```
      Or just paste the whole curl command to an assistant/session that
      has access to this repo and ask it to bootstrap for you.

      (`--cookie-header "<value>" --user-agent "<value>"` still works too,
      if you'd rather extract those two values yourself.)
   5. **Verify it actually works** rather than assume — a copy/paste can
      silently truncate a long cookie value (terminal wrapping, a UI
      ellipsis, etc.), and `bootstrap_session.py` can't detect that on
      its own:
      ```
      python -c "
      import sys; sys.path.insert(0, 'src')
      from darija_forum.http_client import ForumHttpClient, SessionExpiredError
      from pathlib import Path
      client = ForumHttpClient(Path('data/state/session.json'))
      try:
          resp = client.get('https://www.djelfa.info/vb/')
          print('status:', resp.status_code, 'len:', len(resp.text))
      except SessionExpiredError as e:
          print('FAILED:', e)
      "
      ```
      `status: 200` with a `len` in the hundreds of thousands means it's
      good. A `FAILED` here means recopy the cookie — the fix is a fresh
      bootstrap, not troubleshooting the code.

   This writes `data/state/session.json`. It's reusable across many
   scrape runs until Cloudflare invalidates it (see Troubleshooting).

## 1. Discover the subforum tree (one-time)

```
python scripts/discover_forum_tree.py
```

Crawls the forum index and writes `data/state/forum_tree.json` (full
tree, for reference) and `data/state/scrape_targets.json` (flattened
list of scrapeable subforum IDs — step 2's input). djelfa.info renders
its entire subforum tree inline on the index page, so this is fast
(confirmed empirically — recursing into every subforum individually
found zero additional forums beyond the index alone).

Only rerun this if you suspect new subforums were added since the last
run — there's no "check for new subforums" mode, it always does a fresh
full crawl.

## 2. Scrape threads/posts

```
python scripts/run_scrape.py
```

Walks every subforum in `scrape_targets.json`, newest-active-thread
first, scraping each thread's posts (all pages). Writes raw JSONL to
`data/raw/djelfa/<subforum_id>/<thread_id>.jsonl`.

Useful flags:
- `--max-threads-per-subforum N` — cap each subforum to its N
  most-recently-active threads. **Strongly recommended** for anything
  beyond a quick test — some subforums have thousands of threads, and
  the default is unbounded.
- `--max-subforums N` — stop after N subforums this run (e.g. to sample
  broadly across categories rather than exhausting one subforum first).
- `--targets-path path/to/file.json` — scrape a custom list instead of
  the full discovered set (a JSON array of subforum-id strings).

Requests go out back-to-back with no rate-limit delay (removed —
deliberate choice, this is your own site).

Example — a modest first real batch across many subforums:
```
python scripts/run_scrape.py --max-threads-per-subforum 10
```

- **Resumable**: if it stops (session expiry, Ctrl-C, crash), rerun the
  same command later — it picks up exactly where it left off (mid
  post-page for a thread, or mid subforum walk).
- **Raising a cap resumes further**: a subforum that stopped because it
  hit `--max-threads-per-subforum` is *not* marked fully done — rerun
  with a higher (or no) cap and it continues from where it stopped,
  without re-scraping threads it already has. Rerunning with the *same or
  lower* cap is a true no-op (zero requests). Only a subforum walked to
  the very end (no cap, or cap never reached) is marked fully complete —
  that one won't resume with a higher cap since there's nothing left to
  fetch.

## 3. Run the dedup pipeline

```
python scripts/run_pipeline.py
```

Reads every raw file not yet processed, applies near-duplicate filtering
(MinHash/LSH, persisted across runs), and writes the schema-conformant
corpus to `data/processed/batch_<date>.jsonl`. Appends one entry to the
global `data/logs/log.json`, including a **per-subforum breakdown** of
posts retained — useful for seeing empirically which sections turn out
Darija-dense, rather than assuming upfront.

`script` and `darija_confidence` are `null` for now — the language/dialect
filter is a deferred phase (see `PLAN.md`).

## 4. Check the output

```
python -c "import json; [print(json.loads(l)['text']) for l in list(open('data/processed/batch_2026-08-05.jsonl', encoding='utf-8'))[:10]]"
```

or open the file directly. Each document matches the nested schema from
`Project_context.md` (`source_metadata.subforum`, `.thread_title`,
`.thread_url`, `.post_url`, `.author`). Check `data/logs/log.json` for
running totals (`posts_collected`, `posts_retained`) and the
`cumulative_by_subforum` breakdown, toward the 1M → 10M → 50M token
milestones.

## Repeating the cycle

Rerun step 2 with higher caps or more subforums whenever you want more
data, then step 3. Both scripts are additive/incremental —
already-scraped threads and already-processed raw files are skipped
automatically.

## Session notes

- No daily quota like YouTube's — the constraint here is Cloudflare's
  session, not a request budget. No rate-limit delay either — requests go
  out as fast as the server responds.
- **Sessions expire** (observed: well under an hour of active use in
  testing — your mileage may vary, and going faster with no delay may
  shorten that further). When a run stops with `SessionExpiredError`,
  nothing is lost — state is saved incrementally. Repeat step 0's
  bootstrap with a fresh cookie, then rerun the same scrape command; it
  resumes exactly where it stopped.

## Troubleshooting

- **`No session saved` / `SessionMissingError`**: run
  `scripts/bootstrap_session.py` (step 0).
- **`SessionExpiredError` mid-run**: the session expired — refresh it
  (step 0) and rerun the same command to resume.
- **`WARNING: no 'cf_clearance' cookie found`** from
  `bootstrap_session.py`: you copied the wrong request (a subresource,
  not the main document) or the cookie header got truncated when
  pasting — recopy per step 0.
- **`Couldn't parse the curl command`**: the pasted text isn't valid
  shell syntax (a stray quote, or something got mangled in the copy) —
  prefer `--curl-file` over `--curl` for long pastes, it sidesteps most
  shell-quoting issues entirely. As a fallback, extract the two values
  yourself and pass `--cookie-header`/`--user-agent` directly.
- **`data/state/scrape_targets.json not found`**: run
  `scripts/discover_forum_tree.py` first (step 1).
- **A specific thread errors out**: check
  `data/state/crawl_state.json` → `threads.<thread_id>.status`. Rerunning
  `run_scrape.py` will retry anything not `"done"`.

## Version control

This project isn't under git yet — everything here is local-only for
now (deliberate choice, not an oversight).
