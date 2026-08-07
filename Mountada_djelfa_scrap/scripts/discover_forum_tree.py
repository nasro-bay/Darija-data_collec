#!/usr/bin/env python
"""CLI: discover djelfa.info's full subforum tree, writing
data/state/forum_tree.json (full tree, for reference/debugging) and
data/state/scrape_targets.json (flattened list of non-private,
non-category forum_ids — sub-plan 2's input).

Requires a bootstrapped session first — see scripts/bootstrap_session.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from darija_forum._atomic import replace_with_retry  # noqa: E402
from darija_forum.discover import discover_forum_tree  # noqa: E402
from darija_forum.http_client import ForumHttpClient  # noqa: E402
from darija_forum.session import SessionMissingError  # noqa: E402


def _write_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    replace_with_retry(tmp_path, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--max-forums",
        type=int,
        default=None,
        help="Cap total forums considered. Note: djelfa.info's index page renders its "
        "*entire* subforum tree inline (confirmed empirically — recursing into every "
        "individual subforum page found zero additional forums beyond the index alone), "
        "so this only limits how many of the already-seeded forums get an extra "
        "verification visit — it won't shrink the initial index-page result itself.",
    )
    parser.add_argument("--session-path", default=str(ROOT / "data" / "state" / "session.json"))
    args = parser.parse_args()

    try:
        client = ForumHttpClient(Path(args.session_path))
    except SessionMissingError as exc:
        raise SystemExit(str(exc))

    forums = discover_forum_tree(client, max_forums=args.max_forums)

    tree_path = ROOT / "data" / "state" / "forum_tree.json"
    targets_path = ROOT / "data" / "state" / "scrape_targets.json"
    _write_json(forums, tree_path)

    targets = [fid for fid, info in forums.items() if not info["is_category"] and not info["is_private"]]
    _write_json(targets, targets_path)

    categories = sum(1 for info in forums.values() if info["is_category"])
    private = sum(1 for info in forums.values() if info["is_private"])
    print(f"Discovered {len(forums)} forum entries: {categories} categories, "
          f"{len(forums) - categories} subforums ({private} flagged private).")
    print(f"Scrape targets (non-private, non-category): {len(targets)}")
    print(f"Wrote {tree_path}")
    print(f"Wrote {targets_path}")


if __name__ == "__main__":
    main()
