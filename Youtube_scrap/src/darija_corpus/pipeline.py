"""Runs newly-scraped raw comments through clean_text.clean() (dropping
near-empty results) and writes the schema-conformant corpus -- appending
one entry to the single global JSON log (data/logs/log.json) per run.

Near-dup dedup (dedup.py, MinHash/LSH) is NOT run here -- disabled because
the sequential LSH query/insert per comment (see dedup.py's docstring) was
the pipeline's dominant cost at this corpus's scale, and deduped-out rows
may be wanted later for training (e.g. weighting/repetition signal)
instead of being discarded. dedup.py is otherwise untouched and still
fully usable as a standalone pass over data/processed/*.jsonl whenever
dedup is wanted again -- see its module docstring.

MinHash computation is ALSO skipped here now (previously done eagerly per
retained comment, even though the LSH pass itself was disabled) -- on
Windows this pipeline spawns a worker process per CPU core, and each
worker has to import dedup.py's `datasketch` -> `scipy` dependency chain;
at this corpus's scale that was enough concurrent DLL loading to exhaust
the paging file (`ImportError: DLL load failed ... The paging file is too
small`). Since nothing downstream currently reads `dedup_hash`, there's no
value being computed and immediately wasted. `dedup_hash` is now written
as `null` in the schema; a future standalone dedup.py pass over
data/processed/*.jsonl can compute MinHash straight from `text` at that
point (dedup.py itself is unchanged and still fully usable).

Cleaning is parallelized across a process pool (one process per CPU core
by default) -- pure, per-comment, CPU-bound, no shared state, so every
comment in a raw file's batch can be handled independently.
"""
from __future__ import annotations

import json
import os
from datetime import date, datetime, timezone
from multiprocessing import Pool
from pathlib import Path

from . import clean_text, schema
from .state import State


def _clean(text: str) -> str | None:
    """Worker-process entry point (must be a module-level function so it's
    picklable for multiprocessing, including Windows' spawn start method).
    """
    return clean_text.clean(text)


def _append_log(log_path: Path, run_entry: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        with log_path.open("r", encoding="utf-8") as f:
            log = json.load(f)
    else:
        log = {
            "runs": [],
            "cumulative": {
                "videos_scraped": 0,
                "comments_collected": 0,
                "comments_dropped_empty": 0,
                "comments_retained": 0,
            },
        }
    log["runs"].append(run_entry)
    for key in log["cumulative"]:
        log["cumulative"][key] += run_entry[key]

    tmp_path = log_path.with_suffix(log_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    tmp_path.replace(log_path)


def run_pipeline(
    *,
    raw_dir: Path,
    processed_dir: Path,
    state: State,
    lsh_path: Path,  # unused -- dedup is disabled, see module docstring; kept so
                      # callers don't need updating if dedup is re-enabled later
    log_path: Path,
    workers: int | None = None,
) -> dict:
    raw_files = sorted(p for p in raw_dir.glob("*.jsonl") if not state.is_raw_file_processed(p.name))

    comments_collected = 0
    comments_dropped_empty = 0
    comments_retained = 0

    today = date.today().isoformat()
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_path = processed_dir / f"batch_{today}.jsonl"

    num_workers = workers if workers is not None else (os.cpu_count() or 1)

    # state.save() rewrites the whole state.json (videos/channels/handles
    # dicts included, not just pipeline progress) -- with 50k+ raw files
    # that's tens of thousands of full-file rewrites if done every
    # iteration, and was the pipeline's real bottleneck once dedup's
    # sequential LSH cost (which used to dwarf this) was removed. Flushed
    # every STATE_SAVE_INTERVAL files instead, and once more at the end;
    # a crash mid-run re-processes at most that many files on resume.
    STATE_SAVE_INTERVAL = 50

    with processed_path.open("a", encoding="utf-8") as out, Pool(processes=num_workers) as pool:
        for i, raw_file in enumerate(raw_files, start=1):
            comments = []
            with raw_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    comments.append(json.loads(line))

            comments_collected += len(comments)

            # Cleaning + MinHash computation happen in parallel across
            # `pool`; imap (not imap_unordered) preserves input order so
            # results line up positionally with `comments` below.
            texts = [c["text"] for c in comments]
            chunksize = max(1, len(texts) // (num_workers * 4)) if texts else 1
            results = pool.imap(_clean, texts, chunksize=chunksize)

            for comment, cleaned in zip(comments, results):
                if cleaned is None:
                    comments_dropped_empty += 1
                    continue
                doc_id = f"yt_{comment['video_id']}_{comment['comment_id']}"
                doc = schema.build_document(
                    doc_id=doc_id,
                    text=cleaned,
                    video_id=comment["video_id"],
                    channel=comment["channel"],
                    scrape_date=comment["scrape_date"],
                    source_type=comment["source_type"],
                    dedup_hash=None,
                )
                out.write(json.dumps(doc, ensure_ascii=False) + "\n")
                comments_retained += 1

            state.mark_raw_file_processed(raw_file.name)
            if i % STATE_SAVE_INTERVAL == 0:
                state.save()

        state.save()

    run_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "videos_scraped": len(raw_files),
        "comments_collected": comments_collected,
        "comments_dropped_empty": comments_dropped_empty,
        "comments_retained": comments_retained,
    }
    _append_log(log_path, run_entry)
    return run_entry
