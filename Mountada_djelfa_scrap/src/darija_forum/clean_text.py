"""Text cleaning rules for djelfa.info forum posts, derived empirically
from a 1,000-post representative sample — see
Notebooks/02_cleaning_rules_djelfa.ipynb for the prevalence data and
rationale behind each rule (including amendments found by validating this
implementation against that sample), plus Notebooks/03_build_vocabulary.ipynb
for the Unicode-normalization rule (found via the vocabulary's "other"
script bucket). Wired into pipeline.py.

Rules preserve expressive language (elongated letters, emoji) rather than
stripping it outright, per the project's "preserve natural variation"
policy — only excess repetition is collapsed, decorative-only characters
(tatweel) are removed outright, and near-empty results are signaled for
the caller to drop, not silently emptied.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Optional

URL_RE = re.compile(r"(?:https?://\S+|www\.\S+)", re.IGNORECASE)

# vBulletin's plain-text reply-quote wrapper. Strips only the boilerplate
# header phrase — the quoted content underneath stays (it's a repost of an
# earlier post; near-dup dedup handles it elsewhere in the pipeline).
QUOTE_WRAPPER_RE = re.compile(r"اقتباس:\s*\nالمشاركة الأصلية كتبت بواسطة\s+\S+")

# BBCode tags (incl. [QUOTE=user;id]/[/QUOTE] — a second quoting mechanism
# that this same generic tag-stripper already handles, no separate rule
# needed) — decorative/structural markup, no semantic content of its own.
# `(?!URL\])` excludes our own `[URL]` anonymization placeholder (inserted
# later by replace_urls) — without it, this pattern matches its own output
# token shape and would eat the placeholder on any re-application of
# clean() to already-cleaned text (confirmed empirically: validating this
# rule's post-clean prevalence showed 16 spurious "leftover BBCode" hits
# that were all `[URL]` tokens, not real markup). `\s*` before the closing
# bracket tolerates an occasional stray newline inside a malformed tag
# (e.g. "[/QUOTE\n]", seen in real scraped data — a source-formatting
# quirk, not something clean_text.py can fully fix generically).
BBCODE_RE = re.compile(r"\[/?(?!URL\])[a-zA-Z]+(?:=[^\]]*)?\s*\]")

# Arabic tatweel/kashida (U+0640) — pure decorative text-stretching, zero
# semantic content (unlike a genuinely repeated letter), so it's deleted
# outright rather than collapsed like real elongation.
TATWEEL_RE = re.compile(r"ـ+")

# "Who's online" widget stat block that leaked into scraped post content —
# confirmed recurring (not a one-off), 100% boilerplate. Strips from the
# marker phrase to the end of the string, since real content was never
# observed to follow it in the sample.
WIDGET_LEAK_RE = re.compile(r"أكبر تواجد بالمنتدى كان.*", re.DOTALL)

# Arabic diacritics (tachkil/tashkeel: fatha, damma, kasra, sukun, shadda,
# tanwin, etc.) plus the superscript alef (dagger alef). Not a Darija
# feature — casual Darija is written essentially undiacritized; diacritized
# fragments found in real samples (26.9% prevalence, much higher than
# YouTube's 0.6%) were religious/poetic quotes, formal text, or usernames
# bleeding in via the quote wrapper — not deliberate dialectal expression.
# Stripped outright (not collapsed) since they carry no dialectal signal
# and just fragment tokenization (e.g. "السلام" vs "السَّلَام" as distinct
# tokens for the same word).
TACHKIL_RE = re.compile("[ً-ٰٟ]")

_ZWJ = "‍"
_VS16 = "️"  # emoji variation selector

EMOJI_CLASS = (
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\U0001F1E6-\U0001F1FF"
    "☀-➿"
)
EMOJI_CHAR_RE = re.compile(f"[{EMOJI_CLASS}]")
# A full emoji "grapheme" — one emoji, optionally ZWJ-joined to more emoji
# (e.g. "🤷‍♀️" = person + ZWJ + female-sign + variation-selector) — treated
# as a single atomic unit so run-collapsing can't truncate a compound emoji
# mid-sequence and leave a dangling ZWJ (confirmed bug on the YouTube side
# of this codebase, ported here since the emoji logic is shared: an earlier
# version treating "emoji + optional single modifier" as the unit collapsed
# "😂🤷‍♀️" into "😂🤷‍", an orphaned ZWJ).
_EMOJI_GRAPHEME_RE = re.compile(f"[{EMOJI_CLASS}]{_VS16}?(?:{_ZWJ}[{EMOJI_CLASS}]{_VS16}?)*")
EMOJI_RUN_RE = re.compile(f"(?:{_EMOJI_GRAPHEME_RE.pattern}){{3,}}")

# Letters only (excludes digits/underscore) — a repeated letter is
# expressive stretching ("هههه", "اااا"); a repeated digit is just a
# number and shouldn't be touched. Applied *after* URL replacement so
# "www" inside a URL never reaches this check.
ELONGATION_RE = re.compile(r"([^\W\d_])\1{2,}", re.UNICODE)

# Periods get gentler treatment than other punctuation: "..." is a common,
# legitimate rhetorical pause in Arabic prose (confirmed: most of this
# corpus's excessive-punctuation hits were multi-dot ellipses, not spam
# emphasis) — normalize to a standard 3-dot ellipsis rather than collapsing
# like "!!!!"/"؟؟؟؟".
PERIOD_RUN_RE = re.compile(r"\.{2,}")
OTHER_PUNCT_RUN_RE = re.compile(r"([!?؟,،])\1{2,}")  # "،" = Arabic comma, distinct from "," (found on real data)

_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_NON_WORD_RE = re.compile(r"[^\w؀-ۿ]", re.UNICODE)

MIN_RESIDUAL_LETTERS = 2  # below this (after stripping emoji/punctuation), drop the doc


def normalize_unicode_forms(text: str) -> str:
    """NFKC-normalizes compatibility characters to their canonical form --
    catches Arabic Presentation Forms ligatures (found via the vocabulary
    notebook's "other" script bucket) that the plain [؀-ۿ] Arabic
    range doesn't recognize as Arabic at all, e.g. "ﷺ" (a single
    religious-phrase ligature) -> "صلى الله عليه وسلم", "ﷲ" -> "الله",
    "ﻻ" -> "لا". Also normalizes presentation-form letter variants
    (e.g. "ﻣﻦ" -> "من") and full-width digits/Latin to their
    plain equivalents. Verified empirically to leave Arabic-Indic digits,
    ASCII digits, tatweel, ZWJ-joined compound emoji, tachkil, and French
    accented letters unchanged -- only touches compatibility-equivalent
    characters, not the real content those other rules depend on.
    """
    return unicodedata.normalize("NFKC", text)


def strip_widget_leak(text: str) -> str:
    return WIDGET_LEAK_RE.sub("", text)


def strip_quote_wrapper(text: str) -> str:
    return QUOTE_WRAPPER_RE.sub(" ", text)


def strip_bbcode(text: str) -> str:
    return BBCODE_RE.sub(" ", text)


def strip_tatweel(text: str) -> str:
    return TATWEEL_RE.sub("", text)


def strip_tachkil(text: str) -> str:
    return TACHKIL_RE.sub("", text)


def replace_urls(text: str) -> str:
    return URL_RE.sub(" [URL] ", text)


def collapse_elongation(text: str) -> str:
    return ELONGATION_RE.sub(r"\1\1", text)


def normalize_punctuation(text: str) -> str:
    text = PERIOD_RUN_RE.sub("...", text)
    text = OTHER_PUNCT_RUN_RE.sub(r"\1\1", text)
    return text


def collapse_emoji_runs(text: str) -> str:
    def _collapse(match: re.Match) -> str:
        graphemes = _EMOJI_GRAPHEME_RE.findall(match.group(0))
        return "".join(graphemes[:2])

    return EMOJI_RUN_RE.sub(_collapse, text)


def normalize_whitespace(text: str) -> str:
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def residual_letter_count(text: str) -> int:
    no_emoji = EMOJI_CHAR_RE.sub("", text)
    no_punct = _NON_WORD_RE.sub("", no_emoji)
    return len(no_punct.strip())


def clean(text: str) -> Optional[str]:
    """Applies all rules in order and returns the cleaned text, or `None`
    if the result should be dropped (near-empty after cleaning).

    Order matters: Unicode normalization runs first so every other rule
    (regex-based, all matching specific codepoint ranges) sees canonical
    text rather than presentation-form ligatures that wouldn't match them.
    Widget-leak and quote-wrapper stripping run next since they're large
    blocks that would otherwise skew everything downstream; tatweel is
    stripped before the general elongation collapse so it isn't miscounted
    as a "real" elongated letter; URLs are replaced before elongation
    collapsing so "www" doesn't get caught by it.
    """
    text = normalize_unicode_forms(text)
    text = strip_widget_leak(text)
    text = strip_quote_wrapper(text)
    text = strip_bbcode(text)
    text = strip_tatweel(text)
    text = strip_tachkil(text)
    text = replace_urls(text)
    text = collapse_elongation(text)
    text = normalize_punctuation(text)
    text = collapse_emoji_runs(text)
    text = normalize_whitespace(text)

    if residual_letter_count(text) < MIN_RESIDUAL_LETTERS:
        return None
    return text
