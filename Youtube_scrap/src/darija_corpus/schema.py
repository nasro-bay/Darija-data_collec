"""Builds the corpus JSON schema (see Readme.md) for a single document.

`script` and `darija_confidence` are left as `null` placeholders — the
language/dialect filter is deferred to a later phase (see PLAN.md).
"""
from __future__ import annotations


def build_document(
    *,
    doc_id: str,
    text: str,
    video_id: str,
    channel: str,
    scrape_date: str,
    source_type: str,
    dedup_hash: str,
) -> dict:
    return {
        "id": doc_id,
        "text": text,
        "source": "youtube",
        "source_type": source_type,
        "video_id": video_id,
        "channel": channel,
        "scrape_date": scrape_date,
        "script": None,
        "darija_confidence": None,
        "char_count": len(text),
        "token_count": len(text.split()),
        "dedup_hash": dedup_hash,
    }
