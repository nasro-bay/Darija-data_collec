#!/usr/bin/env python
"""Local web CRUD editor for this folder's two eval datasets
(data/word_similarity.jsonl, data/analogy_pairs.jsonl) -- add/delete
entries through a browser instead of hand-editing JSONL. Reads and
writes the real files directly (atomic write: tmp file + os.replace, so
a crash mid-save can't leave a corrupted JSONL behind).

Run: python app.py, then open http://127.0.0.1:5050
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
SIMILARITY_PATH = DATA_DIR / "word_similarity.jsonl"
ANALOGY_PATH = DATA_DIR / "analogy_pairs.jsonl"

SIMILARITY_CATEGORIES = [
    "synonym", "antonym", "cross_script", "code_switch", "morphological_variant", "unrelated",
]
ANALOGY_CATEGORIES = ["gender", "singular_plural", "script_transliteration"]

app = Flask(__name__)


def _atomic_write_lines(path: Path, lines: list[str]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    os.replace(tmp_path, path)


# ------------------------------------------------------------ word_similarity --

def _load_similarity() -> list[dict]:
    if not SIMILARITY_PATH.exists():
        return []
    with SIMILARITY_PATH.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    for i, row in enumerate(rows):
        row["id"] = i
    return rows


def _save_similarity(rows: list[dict]) -> None:
    lines = [
        json.dumps(
            {"word1": r["word1"], "word2": r["word2"], "category": r["category"], "score": r["score"], "note": r.get("note", "")},
            ensure_ascii=False,
        )
        for r in rows
    ]
    _atomic_write_lines(SIMILARITY_PATH, lines)


# -------------------------------------------------------------- analogy_pairs --

def _load_analogy_blocks() -> list[dict]:
    if not ANALOGY_PATH.exists():
        return []
    with ANALOGY_PATH.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _flatten_analogy(blocks: list[dict]) -> list[dict]:
    flat = []
    running_id = 0
    for block in blocks:
        for pair in block["pairs"]:
            flat.append({
                "id": running_id,
                "category": block["category"],
                "word_a": pair["word_a"],
                "word_b": pair["word_b"],
                "note": pair.get("note", ""),
            })
            running_id += 1
    return flat


def _save_analogy_flat(flat: list[dict]) -> None:
    by_category: dict[str, list[dict]] = {}
    for row in flat:
        by_category.setdefault(row["category"], []).append(
            {"word_a": row["word_a"], "word_b": row["word_b"], "note": row.get("note", "")}
        )
    lines = [
        json.dumps({"category": cat, "pairs": pairs}, ensure_ascii=False)
        for cat, pairs in by_category.items()
        if pairs
    ]
    _atomic_write_lines(ANALOGY_PATH, lines)


# ------------------------------------------------------------------- routes --

@app.route("/")
def index():
    return render_template(
        "index.html",
        similarity_categories=SIMILARITY_CATEGORIES,
        analogy_categories=ANALOGY_CATEGORIES,
    )


@app.route("/api/word_similarity", methods=["GET"])
def get_similarity():
    return jsonify(_load_similarity())


@app.route("/api/word_similarity", methods=["POST"])
def add_similarity():
    body = request.get_json(force=True)
    for field in ("word1", "word2", "category", "score"):
        if field not in body or body[field] in (None, ""):
            return jsonify({"error": f"missing field: {field}"}), 400
    try:
        score = float(body["score"])
    except (TypeError, ValueError):
        return jsonify({"error": "score must be a number"}), 400
    if not (0.0 <= score <= 1.0):
        return jsonify({"error": "score must be between 0 and 1"}), 400
    if body["category"] not in SIMILARITY_CATEGORIES:
        return jsonify({"error": f"category must be one of {SIMILARITY_CATEGORIES}"}), 400

    rows = _load_similarity()
    new_row = {
        "word1": body["word1"].strip(),
        "word2": body["word2"].strip(),
        "category": body["category"],
        "score": round(score, 3),
        "note": body.get("note", "").strip(),
    }
    rows.append(new_row)
    _save_similarity(rows)
    new_row["id"] = len(rows) - 1
    return jsonify(new_row), 201


@app.route("/api/word_similarity/<int:row_id>", methods=["DELETE"])
def delete_similarity(row_id: int):
    rows = _load_similarity()
    if row_id < 0 or row_id >= len(rows):
        return jsonify({"error": "not found"}), 404
    del rows[row_id]
    _save_similarity(rows)
    return jsonify({"ok": True})


@app.route("/api/word_similarity/<int:row_id>", methods=["PUT"])
def update_similarity(row_id: int):
    rows = _load_similarity()
    if row_id < 0 or row_id >= len(rows):
        return jsonify({"error": "not found"}), 404

    body = request.get_json(force=True)
    for field in ("word1", "word2", "category", "score"):
        if field not in body or body[field] in (None, ""):
            return jsonify({"error": f"missing field: {field}"}), 400
    try:
        score = float(body["score"])
    except (TypeError, ValueError):
        return jsonify({"error": "score must be a number"}), 400
    if not (0.0 <= score <= 1.0):
        return jsonify({"error": "score must be between 0 and 1"}), 400
    if body["category"] not in SIMILARITY_CATEGORIES:
        return jsonify({"error": f"category must be one of {SIMILARITY_CATEGORIES}"}), 400

    # Flat list, no regrouping on save -- unlike analogy_pairs, the id
    # stays stable across an edit, so this can just overwrite in place.
    rows[row_id] = {
        "word1": body["word1"].strip(),
        "word2": body["word2"].strip(),
        "category": body["category"],
        "score": round(score, 3),
        "note": body.get("note", "").strip(),
    }
    _save_similarity(rows)
    result = dict(rows[row_id])
    result["id"] = row_id
    return jsonify(result)


@app.route("/api/analogy_pairs", methods=["GET"])
def get_analogy():
    return jsonify(_flatten_analogy(_load_analogy_blocks()))


@app.route("/api/analogy_pairs", methods=["POST"])
def add_analogy():
    body = request.get_json(force=True)
    for field in ("word_a", "word_b", "category"):
        if field not in body or body[field] in (None, ""):
            return jsonify({"error": f"missing field: {field}"}), 400
    if body["category"] not in ANALOGY_CATEGORIES:
        return jsonify({"error": f"category must be one of {ANALOGY_CATEGORIES}"}), 400

    flat = _flatten_analogy(_load_analogy_blocks())
    new_row = {
        "category": body["category"],
        "word_a": body["word_a"].strip(),
        "word_b": body["word_b"].strip(),
        "note": body.get("note", "").strip(),
    }
    flat.append(new_row)
    _save_analogy_flat(flat)

    # `len(flat) - 1` (append-order position) is NOT the row's real id --
    # _save_analogy_flat regroups by category, so the new row ends up at
    # the end of *its own category's* block, not the end of the whole
    # file, unless its category happens to be the last one on disk. Any
    # id computed before the regrouping save goes stale the instant the
    # file is written. Re-flatten the just-saved file (the authoritative
    # order) and take the last entry in the new row's category -- that's
    # guaranteed to be the one just added, since nothing else in that
    # category was appended after it.
    fresh_flat = _flatten_analogy(_load_analogy_blocks())
    same_category = [r for r in fresh_flat if r["category"] == new_row["category"]]
    return jsonify(same_category[-1]), 201


@app.route("/api/analogy_pairs/<int:row_id>", methods=["DELETE"])
def delete_analogy(row_id: int):
    flat = _flatten_analogy(_load_analogy_blocks())
    if row_id < 0 or row_id >= len(flat):
        return jsonify({"error": "not found"}), 404
    del flat[row_id]
    _save_analogy_flat(flat)
    return jsonify({"ok": True})


@app.route("/api/analogy_pairs/<int:row_id>", methods=["PUT"])
def update_analogy(row_id: int):
    flat = _flatten_analogy(_load_analogy_blocks())
    if row_id < 0 or row_id >= len(flat):
        return jsonify({"error": "not found"}), 404

    body = request.get_json(force=True)
    for field in ("word_a", "word_b", "category"):
        if field not in body or body[field] in (None, ""):
            return jsonify({"error": f"missing field: {field}"}), 400
    if body["category"] not in ANALOGY_CATEGORIES:
        return jsonify({"error": f"category must be one of {ANALOGY_CATEGORIES}"}), 400

    updated_row = {
        "category": body["category"],
        "word_a": body["word_a"].strip(),
        "word_b": body["word_b"].strip(),
        "note": body.get("note", "").strip(),
    }
    flat[row_id] = updated_row
    _save_analogy_flat(flat)

    # Same staleness issue as add_analogy(): editing can change category
    # (or just reordering on save), so re-derive the id from the
    # just-saved file rather than trusting `row_id` still points to this
    # row after the regrouping save.
    fresh_flat = _flatten_analogy(_load_analogy_blocks())
    same_category = [
        r for r in fresh_flat
        if r["category"] == updated_row["category"]
        and r["word_a"] == updated_row["word_a"]
        and r["word_b"] == updated_row["word_b"]
    ]
    return jsonify(same_category[-1] if same_category else updated_row)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)
