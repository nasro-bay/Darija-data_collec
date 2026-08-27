"""Persisted, resumable state: subforum/thread crawl progress and
pipeline progress. Backed by a single JSON file so scraping and the
pipeline can be interrupted (crash, session expiry) and resumed without
redoing work. Mirrors Youtube_scrap's state.py pattern (including the
capped_at/completed distinction for a boundable-then-resumable walk).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ._atomic import replace_with_retry

_DEFAULT_STATE = {
    "subforums": {},
    "threads": {},
    "pipeline": {"processed_raw_files": []},
}


class State:
    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, Any] = self._load()
        # See Youtube_scrap's state.py -- cached to avoid an O(n) list scan
        # per is_raw_file_processed() call against a growing raw-file count.
        self._processed_raw_files_set: set[str] = set(self.data["pipeline"]["processed_raw_files"])

    def _load(self) -> dict:
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = json.loads(json.dumps(_DEFAULT_STATE))
        for key, default in _DEFAULT_STATE.items():
            data.setdefault(key, json.loads(json.dumps(default)))
        return data

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        replace_with_retry(tmp_path, self.path)

    # --- subforum progress ---
    def subforum_state(self, forum_id: str) -> dict:
        return self.data["subforums"].setdefault(
            forum_id,
            {
                "next_thread_page": 1,
                "threads_found": 0,
                "counted_thread_ids": [],
                "capped_at": None,
                "completed": False,
            },
        )

    # --- thread progress ---
    def thread_state(self, thread_id: str) -> dict:
        return self.data["threads"].setdefault(
            thread_id,
            {"subforum_id": None, "title": None, "next_post_page": 1, "status": "pending"},
        )

    # --- pipeline progress ---
    def is_raw_file_processed(self, filename: str) -> bool:
        return filename in self._processed_raw_files_set

    def mark_raw_file_processed(self, filename: str) -> None:
        self.data["pipeline"]["processed_raw_files"].append(filename)
        self._processed_raw_files_set.add(filename)
