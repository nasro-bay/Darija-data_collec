"""Solves djelfa.info's Cloudflare JS challenge once with a real headless
browser, harvesting a cf_clearance cookie + matching User-Agent that a
plain HTTP client can then reuse for bulk crawling (see `http_client.py`)
— refreshed only when the session expires or a challenge reappears.
"""
from __future__ import annotations

import json
import re
import shlex
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from playwright.sync_api import sync_playwright

from ._atomic import replace_with_retry

FORUM_INDEX_URL = "https://www.djelfa.info/vb/"
CHALLENGE_TITLE_MARKERS = ("just a moment", "attention required", "checking your browser")
CHALLENGE_WAIT_TIMEOUT_S = 30
CHALLENGE_POLL_INTERVAL_S = 0.5


class ChallengeNotSolvedError(RuntimeError):
    """Raised when the Cloudflare challenge doesn't clear within the timeout."""


class SessionMissingError(RuntimeError):
    """Raised when no session is saved and automatic solving isn't viable."""


def looks_like_challenge_title(title: str) -> bool:
    lowered = title.lower()
    return any(marker in lowered for marker in CHALLENGE_TITLE_MARKERS)


def solve_challenge(*, url: str = FORUM_INDEX_URL, headless: bool = True) -> dict:
    """Launches a headless browser, waits for the Cloudflare challenge to
    clear, and returns {"cookies": [...], "user_agent": ..., "solved_at": ...}.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context()
        page = context.new_page()
        page.goto(url, wait_until="domcontentloaded")

        deadline = time.monotonic() + CHALLENGE_WAIT_TIMEOUT_S
        while looks_like_challenge_title(page.title()):
            if time.monotonic() > deadline:
                stuck_title = page.title()
                browser.close()
                raise ChallengeNotSolvedError(
                    f"Cloudflare challenge did not clear within {CHALLENGE_WAIT_TIMEOUT_S}s "
                    f"(stuck on title: {stuck_title!r}). If this persists, the challenge may "
                    "have escalated to an interactive CAPTCHA — that needs a Cloudflare-side "
                    "allowlist rule for this machine's IP, not further automation here."
                )
            page.wait_for_timeout(int(CHALLENGE_POLL_INTERVAL_S * 1000))

        cookies = context.cookies()
        user_agent = page.evaluate("() => navigator.userAgent")
        browser.close()

    return {
        "cookies": cookies,
        "user_agent": user_agent,
        "solved_at": datetime.now(timezone.utc).isoformat(),
    }


def save_session(session: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(session, f, ensure_ascii=False, indent=2)
    replace_with_retry(tmp_path, path)


def load_session(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_or_refresh_session(path: Path) -> dict:
    """Reuses a saved session if present.

    Does NOT fall back to `solve_challenge()` automatically: confirmed
    against djelfa.info that Cloudflare detects Playwright's automation
    fingerprint and never clears the challenge for it (headless or
    headed), so an automatic attempt here would just hang for
    `CHALLENGE_WAIT_TIMEOUT_S` and fail every time. Use
    `scripts/bootstrap_session.py` (manual, from a real browser) instead.
    """
    existing = load_session(path)
    if existing:
        return existing
    raise SessionMissingError(
        f"No session saved at {path}. Run scripts/bootstrap_session.py with a "
        "cf_clearance cookie + User-Agent copied from a real browser session "
        "(see that script's --help)."
    )


def parse_cookie_header(cookie_header: str, *, domain: str = ".djelfa.info", path: str = "/") -> list[dict]:
    """Splits a raw `Cookie:` request-header string ('name=value; name2=value2')
    into the {"name", "value", "domain", "path"} dicts `http_client.py` expects.
    """
    cookies = []
    for part in cookie_header.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        name, value = part.split("=", 1)
        cookies.append({"name": name.strip(), "value": value.strip(), "domain": domain, "path": path})
    return cookies


class CurlParseError(RuntimeError):
    """Raised when a pasted curl command doesn't yield a cookie header + User-Agent."""


def parse_curl_command(curl_command: str) -> dict:
    """Extracts {"cookie_header", "user_agent"} from a curl command copied
    via a browser's DevTools → Network → "Copy as cURL (bash)" — the
    -b/--cookie flag's value and the User-Agent header, ignoring every
    other flag (accept, referer, sec-ch-ua-*, --data-raw, ...), which
    belong to that one specific request, not the reusable session.
    """
    # Shell line-continuations ("\" immediately followed by a newline)
    # need joining into one logical line before tokenizing.
    joined = re.sub(r"\\\r?\n", " ", curl_command)
    try:
        tokens = shlex.split(joined, posix=True)
    except ValueError as exc:
        raise CurlParseError(f"Couldn't parse the curl command as shell syntax: {exc}") from exc

    cookie_header: Optional[str] = None
    user_agent: Optional[str] = None

    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in ("-b", "--cookie") and i + 1 < len(tokens):
            cookie_header = tokens[i + 1]
            i += 2
            continue
        if token in ("-H", "--header") and i + 1 < len(tokens):
            header = tokens[i + 1]
            name, _, value = header.partition(":")
            if name.strip().lower() == "user-agent":
                user_agent = value.strip()
            i += 2
            continue
        if token in ("-A", "--user-agent") and i + 1 < len(tokens):
            user_agent = tokens[i + 1]
            i += 2
            continue
        i += 1

    if cookie_header is None:
        raise CurlParseError("No -b/--cookie flag found in the curl command.")
    if user_agent is None:
        raise CurlParseError("No User-Agent header (-H 'user-agent: ...' or -A) found in the curl command.")

    return {"cookie_header": cookie_header, "user_agent": user_agent}


def save_manual_session(
    *, cookie_header: str, user_agent: str, path: Path, domain: str = ".djelfa.info"
) -> dict:
    """Persists a session harvested manually from a real browser (DevTools →
    Network → the main document request → "Copy as cURL") — the working
    fallback since `solve_challenge()` can't get past Cloudflare's
    automation-fingerprint detection on this site.
    """
    session = {
        "cookies": parse_cookie_header(cookie_header, domain=domain),
        "user_agent": user_agent,
        "solved_at": datetime.now(timezone.utc).isoformat(),
        "source": "manual",
    }
    save_session(session, path)
    return session


def save_session_from_curl(curl_command: str, *, path: Path, domain: str = ".djelfa.info") -> dict:
    """Parses a "Copy as cURL (bash)" command and saves the session in one
    step — see `parse_curl_command()` / `save_manual_session()`.
    """
    parsed = parse_curl_command(curl_command)
    return save_manual_session(
        cookie_header=parsed["cookie_header"], user_agent=parsed["user_agent"], path=path, domain=domain
    )
