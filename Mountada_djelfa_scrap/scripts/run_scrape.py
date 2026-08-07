#!/usr/bin/env python
"""CLI: scrape djelfa.info forum posts for the subforums discovered by
scripts/discover_forum_tree.py (data/state/scrape_targets.json).
Resumable: safe to re-run after a crash or a session expiry (rerun
scripts/bootstrap_session.py first if the session expired).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from darija_forum.http_client import ForumHttpClient, SessionExpiredError  # noqa: E402
from darija_forum.scrape import scrape_subforum  # noqa: E402
from darija_forum.session import SessionMissingError  # noqa: E402
from darija_forum.state import State  # noqa: E402


def load_targets(path: Path) -> list[str]:
    if not path.exists():
        raise SystemExit(f"{path} not found. Run scripts/discover_forum_tree.py first.")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--targets-path", default=str(ROOT / "data" / "state" / "scrape_targets.json"))
    parser.add_argument(
        "--max-threads-per-subforum",
        type=int,
        default=None,
        help="Cap each subforum to its N most-recently-active threads (raise later to resume further).",
    )
    parser.add_argument(
        "--max-subforums", type=int, default=None, help="Stop after this many subforums this run."
    )
    parser.add_argument("--session-path", default=str(ROOT / "data" / "state" / "session.json"))
    parser.add_argument("--state-path", default=str(ROOT / "data" / "state" / "crawl_state.json"))
    args = parser.parse_args()

    targets = load_targets(Path(args.targets_path))

    try:
        client = ForumHttpClient(Path(args.session_path))
    except SessionMissingError as exc:
        raise SystemExit(str(exc))

    state = State(Path(args.state_path))
    raw_dir = ROOT / "data" / "raw" / "djelfa"

    subforums_processed = 0
    for forum_id in targets:
        if args.max_subforums is not None and subforums_processed >= args.max_subforums:
            break
        cap_note = f" (max_threads={args.max_threads_per_subforum})" if args.max_threads_per_subforum else ""
        print(f"Scraping subforum f={forum_id}{cap_note}")
        try:
            result = scrape_subforum(
                client, state, raw_dir, forum_id, max_threads=args.max_threads_per_subforum
            )
            print(f"  -> {result}")
        except SessionExpiredError as exc:
            print(f"  Stopped: {exc}")
            break
        subforums_processed += 1

    print(f"\nSubforums processed this run: {subforums_processed}")


if __name__ == "__main__":
    main()
