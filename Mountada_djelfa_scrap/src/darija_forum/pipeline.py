"""Runs newly-scraped raw posts through clean_text.clean() (dropping
near-empty results), then near-dup dedup on the *cleaned* text, and writes
the schema-conformant corpus, appending one entry to the single global
JSON log (data/logs/log.json) per run — including a per-subforum breakdown
of retained posts, per this source's explicit deliverable (see PLAN.md):
seeing empirically which sections turned out Darija-dense, not assumed
up front. Cleaning runs before dedup so boilerplate variation (quote
wrappers, BBCode, tatweel, etc.) doesn't hide near-duplicates from each
other — e.g. two reposts quoting different users but with identical reply
text look distinct pre-cleaning, identical post-cleaning.

Cleaning + MinHash computation are parallelized across a process pool
(one process per CPU core by default), same rationale and the same
sequential-LSH-check caveat as Youtube_scrap's pipeline.py: both are pure,
per-post, CPU-bound operations with no shared state, but the near-dup LSH
check mutates a single shared index and must stay sequential, in the raw
file's original order, to keep dedup results identical to the
pre-parallelization pipeline.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from datetime import date, datetime, timezone
from multiprocessing import Pool
from pathlib import Path

from . import clean_text, dedup, schema
from .state import State

# How often (in raw files) to persist the MinHash LSH index mid-run, in
# addition to always saving it at the end. Bounds how much dedup state a
# killed process can lose, without the overhead of saving it after every
# single file.
LSH_SAVE_INTERVAL = 200


def _clean_and_hash(text: str) -> tuple[str | None, "dedup.MinHash | None"]:
    """Worker-process entry point (must be a module-level function so it's
    picklable for multiprocessing, including Windows' spawn start method).
    """
    cleaned = clean_text.clean(text)
    if cleaned is None:
        return None, None
    return cleaned, dedup.text_to_minhash(cleaned)


def _append_log(log_path: Path, run_entry: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        with log_path.open("r", encoding="utf-8") as f:
            log = json.load(f)
    else:
        log = {
            "runs": [],
            "cumulative": {
                "threads_processed": 0,
                "posts_collected": 0,
                "posts_dropped_empty": 0,
                "posts_retained": 0,
            },
            "cumulative_by_subforum": {},
        }
    log["runs"].append(run_entry)
    for key in ("threads_processed", "posts_collected", "posts_dropped_empty", "posts_retained"):
        log["cumulative"][key] += run_entry[key]
    for subforum_id, counts in run_entry["by_subforum"].items():
        totals = log["cumulative_by_subforum"].setdefault(
            subforum_id, {"posts_collected": 0, "posts_dropped_empty": 0, "posts_retained": 0}
        )
        totals["posts_collected"] += counts["posts_collected"]
        totals["posts_dropped_empty"] += counts["posts_dropped_empty"]
        totals["posts_retained"] += counts["posts_retained"]

    tmp_path = log_path.with_suffix(log_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    tmp_path.replace(log_path)


def run_pipeline(
    *,
    raw_dir: Path,
    processed_dir: Path,
    state: State,
    lsh_path: Path,
    log_path: Path,
    workers: int | None = None,
) -> dict:
    raw_files = sorted(
        p for p in raw_dir.rglob("*.jsonl")
        if not state.is_raw_file_processed(p.relative_to(raw_dir).as_posix())
    )
    lsh = dedup.load_lsh(lsh_path)

    posts_collected = 0
    posts_dropped_empty = 0
    posts_retained = 0
    by_subforum: Counter = Counter()
    by_subforum_dropped: Counter = Counter()
    by_subforum_retained: Counter = Counter()

    today = date.today().isoformat()
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_path = processed_dir / f"batch_{today}.jsonl"

    num_workers = workers if workers is not None else (os.cpu_count() or 1)

    with Pool(processes=num_workers) as pool:
        for i, raw_file in enumerate(raw_files):
            posts = []
            with raw_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    posts.append(json.loads(line))

            posts_collected += len(posts)
            for post in posts:
                by_subforum[post["subforum_id"]] += 1

            # Cleaning + MinHash computation happen in parallel across
            # `pool`; imap (not imap_unordered) preserves input order so
            # results line up positionally with `posts` below.
            texts = [p["text"] for p in posts]
            chunksize = max(1, len(texts) // (num_workers * 4)) if texts else 1
            results = pool.imap(_clean_and_hash, texts, chunksize=chunksize)

            lines_out = []
            for post, (cleaned, minhash) in zip(posts, results):
                if cleaned is None:
                    posts_dropped_empty += 1
                    by_subforum_dropped[post["subforum_id"]] += 1
                    continue
                doc_id = f"djelfa_{post['post_id']}"
                if dedup.is_near_duplicate(lsh, doc_id, minhash):
                    continue
                doc = schema.build_document(
                    doc_id=doc_id,
                    text=cleaned,
                    subforum=post["subforum_id"],
                    thread_title=post["thread_title"],
                    thread_url=post["thread_url"],
                    post_url=post["post_url"],
                    author=post.get("author"),
                    scrape_date=post["scrape_date"],
                    dedup_hash=dedup.minhash_digest(minhash),
                )
                lines_out.append(json.dumps(doc, ensure_ascii=False))
                posts_retained += 1
                by_subforum_retained[post["subforum_id"]] += 1

            # Open, write, and durably flush per raw file — guarantees that by
            # the time a file is marked "processed" (below), its output is
            # actually on disk. A single file handle kept open across the
            # whole run (the old approach) let a killed process lose already
            # -"processed" files' output silently, since state.json's
            # atomic-replace saves are durable but a plain buffered write
            # isn't until flushed/closed — confirmed: one interrupted run
            # lost ~44,700 of ~51,000 claimed-retained posts this way, with
            # nothing in state.json to reveal it had happened.
            if lines_out:
                with processed_path.open("a", encoding="utf-8") as out:
                    for line in lines_out:
                        out.write(line + "\n")
                    out.flush()
                    os.fsync(out.fileno())

            state.mark_raw_file_processed(raw_file.relative_to(raw_dir).as_posix())
            state.save()

            if (i + 1) % LSH_SAVE_INTERVAL == 0:
                dedup.save_lsh(lsh, lsh_path)

    dedup.save_lsh(lsh, lsh_path)

    threads_processed = len(raw_files)
    run_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "threads_processed": threads_processed,
        "posts_collected": posts_collected,
        "posts_dropped_empty": posts_dropped_empty,
        "posts_retained": posts_retained,
        "by_subforum": {
            subforum_id: {
                "posts_collected": by_subforum[subforum_id],
                "posts_dropped_empty": by_subforum_dropped[subforum_id],
                "posts_retained": by_subforum_retained[subforum_id],
            }
            for subforum_id in by_subforum
        },
    }
    _append_log(log_path, run_entry)
    return run_entry
