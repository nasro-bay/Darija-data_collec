"""Runs newly-scraped raw posts through near-dup dedup and writes the
schema-conformant corpus, appending one entry to the single global JSON
log (data/logs/log.json) per run — including a per-subforum breakdown of
retained posts, per this source's explicit deliverable (see PLAN.md):
seeing empirically which sections turned out Darija-dense, not assumed
up front.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

from . import dedup, schema
from .state import State

# How often (in raw files) to persist the MinHash LSH index mid-run, in
# addition to always saving it at the end. Bounds how much dedup state a
# killed process can lose, without the overhead of saving it after every
# single file.
LSH_SAVE_INTERVAL = 200


def _append_log(log_path: Path, run_entry: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        with log_path.open("r", encoding="utf-8") as f:
            log = json.load(f)
    else:
        log = {
            "runs": [],
            "cumulative": {"threads_processed": 0, "posts_collected": 0, "posts_retained": 0},
            "cumulative_by_subforum": {},
        }
    log["runs"].append(run_entry)
    for key in ("threads_processed", "posts_collected", "posts_retained"):
        log["cumulative"][key] += run_entry[key]
    for subforum_id, counts in run_entry["by_subforum"].items():
        totals = log["cumulative_by_subforum"].setdefault(
            subforum_id, {"posts_collected": 0, "posts_retained": 0}
        )
        totals["posts_collected"] += counts["posts_collected"]
        totals["posts_retained"] += counts["posts_retained"]

    tmp_path = log_path.with_suffix(log_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    tmp_path.replace(log_path)


def run_pipeline(
    *, raw_dir: Path, processed_dir: Path, state: State, lsh_path: Path, log_path: Path
) -> dict:
    raw_files = sorted(
        p for p in raw_dir.rglob("*.jsonl")
        if not state.is_raw_file_processed(p.relative_to(raw_dir).as_posix())
    )
    lsh = dedup.load_lsh(lsh_path)

    posts_collected = 0
    posts_retained = 0
    by_subforum: Counter = Counter()
    by_subforum_retained: Counter = Counter()

    today = date.today().isoformat()
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_path = processed_dir / f"batch_{today}.jsonl"

    for i, raw_file in enumerate(raw_files):
        lines_out = []
        with raw_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                post = json.loads(line)
                posts_collected += 1
                by_subforum[post["subforum_id"]] += 1
                text = post["text"]
                minhash = dedup.text_to_minhash(text)
                doc_id = f"djelfa_{post['post_id']}"
                if dedup.is_near_duplicate(lsh, doc_id, minhash):
                    continue
                doc = schema.build_document(
                    doc_id=doc_id,
                    text=text,
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
        "posts_retained": posts_retained,
        "by_subforum": {
            subforum_id: {
                "posts_collected": by_subforum[subforum_id],
                "posts_retained": by_subforum_retained[subforum_id],
            }
            for subforum_id in by_subforum
        },
    }
    _append_log(log_path, run_entry)
    return run_entry
