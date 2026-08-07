# YouTube Darija Corpus Collector — Implementation Plan

## Context

`Readme.md` defines the project: build a Darija text-pretraining corpus,
starting with YouTube comments as the first source, via the official
YouTube Data API v3. The project directory is currently empty (only the
Readme exists) — this is a greenfield build. The plan below implements the
full pipeline described in the Readme's "Current phase" section: scrape →
filter → dedupe → clean → store → report, using Python (matches the
tooling used in the user's sibling projects).

Two things the Readme leaves open that this plan resolves:
- **Seed videos**: scraping is driven by a user-curated list of specific
  video links/IDs (`config/seed_videos.yaml`), not by walking channel
  upload playlists — the user picks exactly which videos to pull comments
  from. A `discover` mode (search.list, regionCode=DZ, Darija keywords)
  is still available as a browsing aid to help find channels worth
  looking at, but it never feeds videos into the scrape list automatically.
- **API key**: the user must create it themselves (Google Cloud Console →
  enable "YouTube Data API v3" → create API key). This plan wires it in via
  `.env`; it does not and cannot provision the key.

## Project structure

```
Darija/
  requirements.txt
  .env.example                 # YOUTUBE_API_KEY=
  config/
    seed_videos.yaml           # user-curated list of video links/IDs to scrape
  src/darija_corpus/
    __init__.py
    youtube_client.py          # thin wrapper over googleapiclient, retry/backoff, quota tracking
    discover.py                # search.list-based channel discovery (regionCode=DZ + keyword seeds) — browsing aid only
    scrape.py                  # video URL/ID -> channel lookup -> commentThreads (+replies)
    state.py                   # checkpoint/resume state, daily quota counter (persisted JSON)
    dedup.py                   # near-dup only (MinHash/LSH via datasketch)
    pipeline.py                # orchestrates dedup -> JSONL write -> global JSON log
    schema.py                  # dataclass/dict builder matching the Readme's JSON schema
  scripts/
    discover_channels.py       # CLI: run channel discovery, print candidates to browse manually
    run_scrape.py              # CLI: scrape raw comments for configured video links (resumable)
    run_pipeline.py            # CLI: raw -> deduped final JSONL + log update
  data/
    raw/<video_id>.jsonl       # untouched scraped comments, one file per video
    processed/batch_<date>.jsonl   # deduped, schema-conformant output
    state/scrape_state.json    # resume checkpoints + quota usage
    state/minhash_lsh.pkl      # persisted MinHash LSH index, cumulative across runs
    logs/log.json              # single global JSON log, appended each run
```

## Key implementation details

**`youtube_client.py`** — wraps `google-api-python-client`. Every call goes
through a `call()` helper that: increments the persisted daily unit
counter, retries with backoff on transient errors (`5xx`, quota
`403 rateLimitExceeded`), and raises/stops cleanly (saving state) if the
daily 10,000-unit budget would be exceeded.

**`discover.py`** — `search.list(type=channel, regionCode=DZ,
relevanceLanguage=ar, q=<keyword from seed list>)` for keywords like
"جزائري فلوق", "comédie algérienne", "دارجة جزائرية" etc. Prints
channel title/id/description for manual browsing — it's a discovery aid
only, not part of the scrape path: you open a candidate channel yourself
and pick specific video links from it.

**`scrape.py`** — scraping is keyed off individual video links, not
channels. `extract_video_id()` parses any common YouTube URL shape
(`watch?v=`, `youtu.be/`, `/shorts/`, `/embed/`, `/live/`) or a bare video
ID. For each configured video: `videos.list` resolves its channel id
(1 unit, only once per video — cached in state) → `commentThreads.list`
(`part=snippet,replies`, 100/page, 1 unit/page) pulls top-level comments
and their first replies, and `comments.list` fetches replies beyond what
was inlined. Writes one raw JSONL file per video immediately (so a crash
doesn't lose prior videos). No separate video-metadata record is kept —
video info (`video_id`, `channel`, `scrape_date`) is only carried as
fields directly on each comment object, per the schema. Skips videos with
comments disabled (catches the 403) and logs them.

**`state.py`** — `data/state/scrape_state.json` tracks: units used today
(keyed by date, resets automatically), each video's resolved channel id,
scrape status, and in-progress comment-page token — so `run_scrape.py` is
safely re-runnable and picks up where it left off, whether interrupted by
quota exhaustion or a crash.

**`dedup.py`** — near-duplicate detection only (no exact-hash pass): each
comment's text is MinHashed and checked against a persisted MinHash LSH
index (`datasketch`, stored at `data/state/minhash_lsh.pkl`) at a
configurable Jaccard threshold. The index is cumulative across runs, so a
repost seen in an earlier batch still gets caught later. `dedup_hash` in
the schema is filled with the MinHash digest used for the comparison.

**`pipeline.py`** — reads all new `data/raw/*.jsonl`, runs dedup, builds
the final schema object per document (`schema.py`), appends to
`data/processed/batch_<date>.jsonl`, and appends a run entry to the single
global `data/logs/log.json` (created if absent) with: run timestamp,
videos scraped, comments collected, comments retained after near-dup
filtering, and running cumulative totals across all runs to date.

**`schema.py`** — builds the JSON object from the Readme (`id`, `text`,
`source`, `source_type`, `video_id`, `channel`, `scrape_date`, `script`,
`darija_confidence`, `char_count`, `token_count`, `dedup_hash`). Since the
language/dialect filter is deferred (see below), `script` and
`darija_confidence` are left as `null` placeholders for now — to be filled
in once that stage is built.

## Deferred (not in this pass, per your request)

- **Darija heuristic filter / script detection** (function-word scoring,
  Arabizi-numeral regex, arabic/latin/mixed bucketing) — dropped from this
  build. Comments are stored unfiltered by language.
- **Cleaning stage** (boilerplate/spam stripping, PII/anonymization
  regexes, min-length filter, mojibake fixes) — dropped from this build.
  Raw comment text flows through to `processed/` as-is aside from dedup.

Both can be added back as their own modules plugged into `pipeline.py`
once you're ready for that phase.

## Dependencies (`requirements.txt`)

`google-api-python-client`, `python-dotenv`, `datasketch`, `pyyaml`,
`tqdm`.

## CLI usage (end state)

```
python scripts/discover_channels.py            # optional: find channels worth browsing for video links
python scripts/run_scrape.py                    # scrape configured seed videos (resumable)
python scripts/run_pipeline.py                   # dedupe -> final JSONL + log update
```

## What this plan does NOT do

- Does not run the scraper or consume API quota — that requires the
  user's own API key and is left for them to trigger.
- Does not hardcode a guessed list of real Algerian video links;
  `config/seed_videos.yaml` ships with the format documented and a
  couple of placeholder/example entries clearly marked for the user to
  replace, plus the `discover` command to help find channels worth
  browsing for links.
- No language/dialect filtering, no cleaning/anonymization stage (see
  Deferred, above), no trained dialect classifier, no audio/speech, no
  other sources.

## Verification

- `python scripts/run_scrape.py --help` / `run_pipeline.py --help` run
  without errors (import sanity check).
- Unit-test `dedup.py`'s near-dup logic against a handful of hand-written
  near-duplicate comment pairs to confirm the MinHash/LSH threshold
  behaves sensibly (no live API calls needed).
- Full end-to-end run is the user's responsibility once they supply an API
  key and populate `config/seed_videos.yaml` — this plan builds the
  pipeline, it doesn't execute a live scrape.

---

# Phase 2: Channel-based scraping (newest videos first)

## Context

Pasting individual video links (Phase 1) doesn't scale to "scrape
everything a channel has." This phase adds a second, coexisting scrape
mode: point at a channel, and it walks that channel's videos
newest-first, feeding each one through the exact same per-video comment
scraper already built (`scrape_video_comments` — no duplicated logic).

**How "newest first" is satisfied for free**: a channel's auto-generated
uploads playlist (the one `channels.list(part=contentDetails)` exposes as
`relatedPlaylists.uploads`) is returned by `playlistItems.list` in
reverse-chronological order — newest upload first — same order as the
channel's "Videos" tab. That means no need for the expensive
`search.list(order=date)` (100 units/call) — plain `playlistItems.list`
(1 unit per 50-video page) already gives newest-first for free.

## What's added

**`youtube_client.py`** — two methods reintroduced (removed in the Phase-1
video-only refactor, now needed again):
- `resolve_channel(*, channel_id=None, handle=None) -> dict` — single
  `channels.list(part="contentDetails", id=... | forHandle=...)` call
  (1 unit) returning `{"channel_id", "uploads_playlist_id"}`.
- `list_playlist_video_ids(playlist_id, page_token) -> (video_ids, next_token)`
  — `playlistItems.list(part="contentDetails", maxResults=50, ...)`
  (1 unit/page).

**`state.py`** — adds back a `channels` dict, keyed by channel id:
`{"uploads_playlist_id", "next_playlist_page_token", "videos_found", "completed"}`.
`videos_found` is a running count used to enforce a per-channel video cap
(see below) across resumed runs. Per-video state (status, comment page
token, channel id) is unchanged and fully shared with Phase 1 — a video
reached via a channel walk that was already scraped via a direct link (or
vice versa) is detected as `"done"` and skipped for free.

**`scrape.py`** — new `scrape_channel(client, state, raw_dir, *, channel_id=None, handle=None, max_videos=None) -> dict`:
1. Resolve channel + uploads playlist id (cached in state after first call).
2. Page through `list_playlist_video_ids` (already newest-first).
3. For each video id: if `max_videos` is set and `videos_found >= max_videos`,
   stop paging and mark the channel `completed`. Otherwise increment
   `videos_found` and call the existing `scrape_video_comments(...)` —
   identical per-video path as Phase 1, so it inherits comments-disabled
   handling, resumable comment pagination, and raw JSONL writing as-is.
4. Persists `next_playlist_page_token` + `videos_found` after every page,
   so an interrupted channel walk resumes mid-pagination on rerun.
5. Returns `{"channel", "videos_considered", "videos_done", "videos_skipped"}`.

**`config/seed_channels.yaml`** (new file, reintroduced with a tweak):
```yaml
channels:
  - handle: "@example_channel"
    max_videos: 20   # optional — omit to eventually walk the whole channel
  - channel_id: "UCxxxxxxxxxxxxxxxxxxxxxx"
```
`max_videos` caps how many of the *newest* videos get scraped — useful to
bound quota spend on channels with huge back catalogs. Omitting it means
"eventually scrape the whole channel," safely spread across as many
resumed runs as needed.

**`scripts/run_scrape_channels.py`** (new CLI, mirrors `run_scrape.py`):
loads `config/seed_channels.yaml`, loops channels calling `scrape_channel`,
catches `QuotaExceededError` the same way (stop, save state, message to
rerun later), prints a per-channel summary and quota used.

`scripts/run_scrape.py` (video-links) is untouched — the two scrape modes
run as separate commands against separate config files, since "scrape
these exact videos" and "scrape whatever a channel has" are different
intents worth keeping explicit rather than silently merged.

`discover_channels.py` stays as-is — still useful for finding
`channel_id`s to paste into the new `seed_channels.yaml`.

`youtube_scrap.md` gets a new section documenting
`run_scrape_channels.py` alongside the existing video-link flow.

## Quota shape (why this stays cheap)

Per channel: 1 unit (resolve) + 1 unit per 50 videos (paging) + the same
per-video comment cost as Phase 1 (~1 unit per 100 comments). A channel
with 200 videos averaging modest comment volume costs roughly 200-400
units total — comfortably resumable across multiple days within the
10,000/day budget if a channel is larger than that.

## Deferred (not in this pass)

- No periodic "check for new videos since last run" refresh mode — once a
  channel is marked `completed` (fully walked, or `max_videos` reached),
  rerunning won't re-check for newer uploads. Can be added later as an
  explicit `--refresh` flag if wanted.

## Verification

- `python scripts/run_scrape_channels.py --help` runs without errors.
- Manual live check: configure one small/low-volume real channel with
  `max_videos: 2`, run it, confirm exactly 2 newest videos get scraped
  into `data/raw/`, confirm a rerun is a no-op (channel already
  `completed`), then run `run_pipeline.py` as usual.
