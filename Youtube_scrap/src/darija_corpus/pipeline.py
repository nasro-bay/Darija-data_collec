"""Runs newly-scraped raw comments through near-dup dedup and writes the
schema-conformant corpus, appending one entry to the single global JSON
log (data/logs/log.json) per run.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

from . import dedup, schema
from .state import State


def _append_log(log_path: Path, run_entry: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        with log_path.open("r", encoding="utf-8") as f:
            log = json.load(f)
    else:
        log = {
            "runs": [],
            "cumulative": {"videos_scraped": 0, "comments_collected": 0, "comments_retained": 0},
        }
    log["runs"].append(run_entry)
    for key in log["cumulative"]:
        log["cumulative"][key] += run_entry[key]

    tmp_path = log_path.with_suffix(log_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    tmp_path.replace(log_path)


def run_pipeline(
    *, raw_dir: Path, processed_dir: Path, state: State, lsh_path: Path, log_path: Path
) -> dict:
    raw_files = sorted(p for p in raw_dir.glob("*.jsonl") if not state.is_raw_file_processed(p.name))
    lsh = dedup.load_lsh(lsh_path)

    comments_collected = 0
    comments_retained = 0

    today = date.today().isoformat()
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_path = processed_dir / f"batch_{today}.jsonl"

    with processed_path.open("a", encoding="utf-8") as out:
        for raw_file in raw_files:
            with raw_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    comment = json.loads(line)
                    comments_collected += 1
                    text = comment["text"]
                    minhash = dedup.text_to_minhash(text)
                    doc_id = f"yt_{comment['video_id']}_{comment['comment_id']}"
                    if dedup.is_near_duplicate(lsh, doc_id, minhash):
                        continue
                    doc = schema.build_document(
                        doc_id=doc_id,
                        text=text,
                        video_id=comment["video_id"],
                        channel=comment["channel"],
                        scrape_date=comment["scrape_date"],
                        source_type=comment["source_type"],
                        dedup_hash=dedup.minhash_digest(minhash),
                    )
                    out.write(json.dumps(doc, ensure_ascii=False) + "\n")
                    comments_retained += 1
            state.mark_raw_file_processed(raw_file.name)
            state.save()

    dedup.save_lsh(lsh, lsh_path)

    run_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "videos_scraped": len(raw_files),
        "comments_collected": comments_collected,
        "comments_retained": comments_retained,
    }
    _append_log(log_path, run_entry)
    return run_entry
