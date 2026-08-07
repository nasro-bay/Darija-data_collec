"""Unit tests for session.py's parse_curl_command() — parsing a browser's
"Copy as cURL (bash)" output into {cookie_header, user_agent}.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from darija_forum.session import CurlParseError, parse_curl_command  # noqa: E402

# A representative multi-line "Copy as cURL (bash)" paste: line
# continuations, several irrelevant headers, a --data-raw challenge-token
# body that must be ignored, and the two values that actually matter.
SAMPLE_CURL = r"""curl 'https://www.djelfa.info/vb/' \
  -H 'accept: text/html,application/xhtml+xml' \
  -H 'accept-language: ar,en-GB;q=0.9' \
  -H 'cache-control: max-age=0' \
  -b 'bbsessionhash=abc123; cf_clearance=XYZ789.token-value_here; _ga=GA1.1.111' \
  -H 'origin: https://www.djelfa.info' \
  -H 'referer: https://www.djelfa.info/vb/?__cf_chl_tk=sometoken' \
  -H 'user-agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36' \
  --data-raw 'sometoken=abcdef-1785958677-1.2.1.1-somechallengevalue'
"""


class ParseCurlCommandTests(unittest.TestCase):
    def test_extracts_cookie_header_and_user_agent(self):
        result = parse_curl_command(SAMPLE_CURL)
        self.assertEqual(
            result["cookie_header"], "bbsessionhash=abc123; cf_clearance=XYZ789.token-value_here; _ga=GA1.1.111"
        )
        self.assertEqual(
            result["user_agent"],
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36",
        )

    def test_ignores_unrelated_flags_and_data_raw(self):
        result = parse_curl_command(SAMPLE_CURL)
        self.assertNotIn("sometoken", result["cookie_header"])
        self.assertNotIn("challenge", result["user_agent"])

    def test_single_line_curl_also_works(self):
        single_line = (
            "curl 'https://www.djelfa.info/vb/' -b 'a=1; cf_clearance=xyz' "
            "-H 'user-agent: TestAgent/1.0'"
        )
        result = parse_curl_command(single_line)
        self.assertEqual(result["cookie_header"], "a=1; cf_clearance=xyz")
        self.assertEqual(result["user_agent"], "TestAgent/1.0")

    def test_long_form_flags(self):
        long_form = "curl 'https://x' --cookie 'a=1' --header 'User-Agent: TestAgent/1.0'"
        result = parse_curl_command(long_form)
        self.assertEqual(result["cookie_header"], "a=1")
        self.assertEqual(result["user_agent"], "TestAgent/1.0")

    def test_dash_a_user_agent_flag(self):
        result = parse_curl_command("curl 'https://x' -b 'a=1' -A 'TestAgent/1.0'")
        self.assertEqual(result["user_agent"], "TestAgent/1.0")

    def test_missing_cookie_raises(self):
        with self.assertRaises(CurlParseError):
            parse_curl_command("curl 'https://x' -H 'user-agent: TestAgent/1.0'")

    def test_missing_user_agent_raises(self):
        with self.assertRaises(CurlParseError):
            parse_curl_command("curl 'https://x' -b 'a=1'")


if __name__ == "__main__":
    unittest.main()
