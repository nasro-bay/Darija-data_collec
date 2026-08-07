"""Persisted, resumable state: daily quota usage, channel/video scrape
progress, and pipeline progress. Backed by a single JSON file so scraping
and the pipeline can be interrupted (crash, quota exhaustion) and resumed
without redoing work.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from ._atomic import replace_with_retry

DAILY_QUOTA_BUDGET = 10_000

_DEFAULT_STATE = {
    "quota": {"date": "", "units_used": 0},
    "videos": {},
    "channels": {},
    "handles": {},
    "pipeline": {"processed_raw_files": []},
}


class QuotaExceededError(RuntimeError):
    """Raised when a call would exceed the daily YouTube API quota budget."""


class State:
    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, Any] = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = json.loads(json.dumps(_DEFAULT_STATE))
        for key, default in _DEFAULT_STATE.items():
            data.setdefault(key, json.loads(json.dumps(default)))
        today = date.today().isoformat()
        if data["quota"].get("date") != today:
            data["quota"] = {"date": today, "units_used": 0}
        return data

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        replace_with_retry(tmp_path, self.path)

    # --- quota ---
    def spend(self, units: int) -> None:
        used = self.data["quota"]["units_used"]
        if used + units > DAILY_QUOTA_BUDGET:
            raise QuotaExceededError(
                f"Would exceed daily quota budget ({used}+{units} > {DAILY_QUOTA_BUDGET})"
            )
        self.data["quota"]["units_used"] = used + units

    def units_remaining(self) -> int:
        return DAILY_QUOTA_BUDGET - self.data["quota"]["units_used"]

    # --- video progress ---
    def video_state(self, video_id: str) -> dict:
        return self.data["videos"].setdefault(
            video_id,
            {"status": "pending", "next_comment_page_token": None, "channel_id": None},
        )

    # --- channel progress ---
    def channel_state(self, channel_id: str) -> dict:
        return self.data["channels"].setdefault(
            channel_id,
            {
                "uploads_playlist_id": None,
                "next_playlist_page_token": None,
                "videos_found": 0,
                "counted_video_ids": [],
                "capped_at": None,
                "completed": False,
            },
        )

    # --- handle -> channel id cache ---
    def resolved_channel_id_for_handle(self, handle: str) -> Any:
        return self.data["handles"].get(handle)

    def remember_handle(self, handle: str, channel_id: str) -> None:
        self.data["handles"][handle] = channel_id

    # --- pipeline progress ---
    def is_raw_file_processed(self, filename: str) -> bool:
        return filename in self.data["pipeline"]["processed_raw_files"]

    def mark_raw_file_processed(self, filename: str) -> None:
        self.data["pipeline"]["processed_raw_files"].append(filename)
