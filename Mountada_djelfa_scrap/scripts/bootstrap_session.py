#!/usr/bin/env python
"""CLI: bootstrap a djelfa.info session from a real browser, since
Cloudflare detects Playwright's automation fingerprint and never clears
the challenge for it (see src/darija_forum/session.py).

How to get what this needs:
  1. Open https://www.djelfa.info/vb/ in a normal browser and let it load.
  2. DevTools (F12) -> Network tab -> reload the page.
  3. Click the main document request (Type: document, near the top) ->
     right-click -> Copy -> Copy as cURL (bash).
  4. Pass that whole curl command to this script (--curl, --curl-file, or
     stdin — see below), or extract --cookie-header/--user-agent yourself.

The session is reusable until Cloudflare invalidates it — rerun this
script to refresh it if scraping starts hitting SessionExpiredError again.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from darija_forum.session import CurlParseError, save_manual_session, save_session_from_curl  # noqa: E402


def _report(session_path: str, session: dict) -> None:
    cookie_names = [c["name"] for c in session["cookies"]]
    print(f"Saved session to {session_path}")
    print(f"Cookies captured ({len(cookie_names)}): {', '.join(cookie_names)}")
    if "cf_clearance" not in cookie_names:
        print("WARNING: no 'cf_clearance' cookie found — this session likely won't pass Cloudflare.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--curl",
        help="The whole 'Copy as cURL (bash)' command as one string. "
        "Cookie header and User-Agent are extracted automatically.",
    )
    parser.add_argument(
        "--curl-file",
        help="Path to a file containing the curl command (avoids shell-quoting a huge "
        "multi-line string) — write your paste to a file and point this at it.",
    )
    parser.add_argument(
        "--cookie-header",
        help="The raw Cookie header value directly (everything after -b), if not using --curl.",
    )
    parser.add_argument(
        "--user-agent",
        help="The exact User-Agent header value directly, if not using --curl.",
    )
    parser.add_argument(
        "--session-path",
        default=str(ROOT / "data" / "state" / "session.json"),
        help="Where to save the session (default: data/state/session.json).",
    )
    args = parser.parse_args()

    session_path = Path(args.session_path)

    if args.curl or args.curl_file:
        curl_text = args.curl if args.curl else Path(args.curl_file).read_text(encoding="utf-8")
        try:
            session = save_session_from_curl(curl_text, path=session_path)
        except CurlParseError as exc:
            raise SystemExit(f"Couldn't parse the curl command: {exc}")
    elif args.cookie_header and args.user_agent:
        session = save_manual_session(cookie_header=args.cookie_header, user_agent=args.user_agent, path=session_path)
    elif not sys.stdin.isatty():
        curl_text = sys.stdin.read()
        try:
            session = save_session_from_curl(curl_text, path=session_path)
        except CurlParseError as exc:
            raise SystemExit(f"Couldn't parse the curl command from stdin: {exc}")
    else:
        raise SystemExit(
            "Provide one of: --curl \"<command>\", --curl-file <path>, "
            "--cookie-header + --user-agent, or pipe the curl command via stdin."
        )

    _report(args.session_path, session)


if __name__ == "__main__":
    main()
