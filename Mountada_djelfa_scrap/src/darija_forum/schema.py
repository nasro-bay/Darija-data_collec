"""Builds the corpus JSON schema (see Project_context.md) for a single
djelfa.info forum post. Uses the nested `source_metadata` schema — the
current project-wide spec, which differs from Youtube_scrap's flat schema
(written before that convention existed; reconciling that is separate,
later cleanup).

`script` and `darija_confidence` are left as `null` placeholders — the
language/dialect filter is deferred to a later phase, same as Youtube_scrap.
"""
from __future__ import annotations

from typing import Optional


def build_document(
    *,
    doc_id: str,
    text: str,
    subforum: str,
    thread_title: str,
    thread_url: str,
    post_url: str,
    author: Optional[str],
    scrape_date: str,
    dedup_hash: str,
) -> dict:
    return {
        "id": doc_id,
        "text": text,
        "source": "djelfa_info",
        "source_type": "forum_post",
        "source_metadata": {
            "subforum": subforum,
            "thread_title": thread_title,
            "thread_url": thread_url,
            "post_url": post_url,
            "author": author,
        },
        "scrape_date": scrape_date,
        "script": None,
        "darija_confidence": None,
        "char_count": len(text),
        "token_count": len(text.split()),
        "dedup_hash": dedup_hash,
    }
