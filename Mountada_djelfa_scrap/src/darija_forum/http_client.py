"""Plain-HTTP client for bulk crawling, reusing a djelfa.info session
harvested by `session.py`. Detects a renewed Cloudflare challenge
(expired session) and raises rather than silently parsing the
interstitial HTML as real page content.
"""
from __future__ import annotations

from pathlib import Path

import requests

from .session import CHALLENGE_TITLE_MARKERS, get_or_refresh_session

REQUEST_TIMEOUT_SECONDS = 30


class SessionExpiredError(RuntimeError):
    """Raised when a request comes back as a Cloudflare challenge page."""


def looks_like_challenge_response(resp: requests.Response) -> bool:
    if resp.status_code == 403:
        return True
    lowered = resp.text[:2000].lower()
    return any(marker in lowered for marker in CHALLENGE_TITLE_MARKERS)


class ForumHttpClient:
    def __init__(self, session_path: Path):
        self.session_path = session_path
        self._session = requests.Session()
        self._load_session_state()

    def _load_session_state(self) -> None:
        session_data = get_or_refresh_session(self.session_path)
        self._session.headers.update(
            {
                "User-Agent": session_data["user_agent"],
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "ar,en-GB;q=0.9,en-US;q=0.8,en;q=0.7,fr;q=0.6",
            }
        )
        for cookie in session_data["cookies"]:
            self._session.cookies.set(
                cookie["name"], cookie["value"], domain=cookie.get("domain"), path=cookie.get("path", "/")
            )

    def get(self, url: str, **kwargs) -> requests.Response:
        resp = self._session.get(url, timeout=REQUEST_TIMEOUT_SECONDS, **kwargs)
        if looks_like_challenge_response(resp):
            raise SessionExpiredError(
                f"Cloudflare challenge reappeared for {url} (status {resp.status_code}) — "
                f"the saved session expired. Rerun scripts/bootstrap_session.py with a fresh "
                f"cookie/User-Agent from a real browser."
            )
        return resp
