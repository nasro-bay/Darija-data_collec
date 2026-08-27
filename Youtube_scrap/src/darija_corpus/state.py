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
        # `processed_raw_files` is a JSON list (order doesn't matter, but a
        # list is what round-trips through json.dump/load), so membership
        # checks against it would otherwise be O(n) -- cached as a set here
        # since is_raw_file_processed() is called once per raw file on every
        # pipeline run, against a raw-file count in the tens of thousands.
        self._processed_raw_files_set: set[str] = set(self.data["pipeline"]["processed_raw_files"])

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

    # --- channel info storage ---
    def record_channel_info(self, channel_id: str, title: str, custom_url: str = "") -> None:
        names_file = self.path.parent / "channel_names.json"
        data = {}
        if names_file.exists():
            try:
                with names_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                pass
        if channel_id not in data or title:
            entry = data.get(channel_id, {"channel_id": channel_id})
            entry["title"] = title or entry.get("title", "")
            entry["custom_url"] = custom_url or entry.get("custom_url", "")
            data[channel_id] = entry
            names_file.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = names_file.with_suffix(".tmp")
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            replace_with_retry(tmp_path, names_file)

    # --- pipeline progress ---
    def is_raw_file_processed(self, filename: str) -> bool:
        return filename in self._processed_raw_files_set

    def mark_raw_file_processed(self, filename: str) -> None:
        self.data["pipeline"]["processed_raw_files"].append(filename)
        self._processed_raw_files_set.add(filename)
