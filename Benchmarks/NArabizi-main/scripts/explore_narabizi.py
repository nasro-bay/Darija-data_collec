#!/usr/bin/env python
"""Parses the NArabizi treebank/sentiment/topic files under ../data/ and
computes the stats behind README.md's write-up: split sizes, token-level
code-switching (from the treebank's own `lang=` annotation -- no
heuristic script-guessing needed here, unlike Twifil), label
distributions, and the ar_dz/ar_na duplication found while exploring this
by hand.

This is a plain stdlib CoNLL-U reader, not a full parser -- it only pulls
out what's needed for the stats below (sent_id, text, per-token lang
tags), not dependency/morphology fields.

Writes ../data/narabizi_report.json (gitignored, same pattern as
Benchmarks/Twifil/data/). Rerun if the data/ folder changes.
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # Benchmarks/NArabizi-main/
DATA_DIR = ROOT / "data"

LANG_TAG_RE = re.compile(r"lang=([a-zA-Z_]+)")


def parse_conllu(path: Path) -> list[dict]:
    """Returns one dict per sentence: {sent_id, text, trad_fr, lang_tags}.
    `lang_tags` is the Counter of each token's `lang=` MISC-field value
    (multi-word tokens like "5-6 fiparout" are skipped -- only the
    single-token sub-rows carry a lang tag in this data).
    """
    sentences = []
    sent_id = text = trad_fr = None
    lang_tags: Counter[str] = Counter()

    def flush():
        if sent_id is not None:
            sentences.append(
                {"sent_id": sent_id, "text": text, "trad_fr": trad_fr, "lang_tags": dict(lang_tags)}
            )

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                flush()
                sent_id = text = trad_fr = None
                lang_tags = Counter()
                continue
            if line.startswith("# sent_id ="):
                sent_id = line.split("=", 1)[1].strip()
            elif line.startswith("# text ="):
                text = line.split("=", 1)[1].strip()
            elif line.startswith("# trad_fr ="):
                trad_fr = line.split("=", 1)[1].strip()
            elif not line.startswith("#"):
                cols = line.split("\t")
                if "-" in cols[0]:  # multi-word token grouping row, not a real token
                    continue
                m = LANG_TAG_RE.search(cols[-1]) if cols else None
                if m:
                    lang_tags[m.group(1)] += 1
    flush()
    return sentences


def load_label_file(path: Path) -> dict[str, str]:
    """Narabizi/sentiment|topic/*.txt: tab-separated `# sent_id = X\t<label>`."""
    labels = {}
    with path.open("r", encoding="utf-8") as f:
        next(f)  # header row ("ID\tSENT" / "ID\tTopic")
        for line in f:
            line = line.rstrip("\n")
            if not line or "\t" not in line:
                continue
            id_part, label = line.rsplit("\t", 1)
            sent_id = id_part.split("=", 1)[1].strip() if "=" in id_part else id_part.strip()
            labels[sent_id] = label
    return labels


def load_binary_sentiment_csv(path: Path) -> list[tuple[str, str]]:
    """sentiment/ar_dz|ar_na/*.csv: `label,text` (text may itself contain
    commas, so this is genuine CSV, not a naive split)."""
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                rows.append((row[0], ",".join(row[1:])))
    return rows


def main() -> None:
    report: dict = {}

    # --- 1. NArabizi treebank (train/dev/test): split sizes + code-switching ---
    pos_dir = DATA_DIR / "Narabizi" / "pos"
    treebank = {}
    for split in ("train", "dev", "test"):
        sentences = parse_conllu(pos_dir / f"{split}_NArabizi.conllu")
        treebank[split] = sentences
        token_lang_counts: Counter[str] = Counter()
        for s in sentences:
            token_lang_counts.update(s["lang_tags"])
        report[f"treebank_{split}_sentences"] = len(sentences)
        report[f"treebank_{split}_lang_tag_distribution"] = dict(token_lang_counts.most_common())

    # --- 2. ud/ar_dz vs ud/ar_na: confirm same content, different FORM/LEMMA script order ---
    ud_dz = parse_conllu(DATA_DIR / "ud" / "ar_dz" / "train.conllu")
    ud_na = parse_conllu(DATA_DIR / "ud" / "ar_na" / "train.conllu")
    report["ud_ar_dz_train_sentences"] = len(ud_dz)
    report["ud_ar_na_train_sentences"] = len(ud_na)
    report["ud_ar_dz_vs_treebank_pos_train_sentence_count_diff"] = len(ud_dz) - len(treebank["train"])

    # --- 3. Narabizi/sentiment + Narabizi/topic: label distributions, joined by sent_id ---
    for kind in ("sentiment", "topic"):
        kind_dir = DATA_DIR / "Narabizi" / kind
        all_labels: Counter[str] = Counter()
        per_split = {}
        for split in ("train", "dev", "test"):
            suffix = "sentiment" if kind == "sentiment" else "topic"
            labels = load_label_file(kind_dir / f"{split}_Narabizi_{suffix}.txt")
            per_split[split] = len(labels)
            all_labels.update(labels.values())
        report[f"narabizi_{kind}_labeled_sentence_count_per_split"] = per_split
        report[f"narabizi_{kind}_label_distribution"] = dict(all_labels.most_common())

    # --- 4. sentiment/ar_dz + ar_na: binary sentiment CSVs, confirm same content two scripts ---
    ar_dz_train = load_binary_sentiment_csv(DATA_DIR / "sentiment" / "ar_dz" / "train.csv")
    ar_na_train = load_binary_sentiment_csv(DATA_DIR / "sentiment" / "ar_na" / "train.csv")
    report["sentiment_ar_dz_ar_na_train_row_count"] = {"ar_dz": len(ar_dz_train), "ar_na": len(ar_na_train)}
    report["sentiment_ar_dz_ar_na_same_labels"] = [l for l, _ in ar_dz_train] == [l for l, _ in ar_na_train]
    label_counts: Counter[str] = Counter()
    for split in ("train", "dev", "test"):
        for label, _ in load_binary_sentiment_csv(DATA_DIR / "sentiment" / "ar_dz" / f"{split}.csv"):
            label_counts[label] += 1
    report["sentiment_ar_dz_label_distribution"] = dict(label_counts)

    # --- 5. Auxiliary multilingual sentiment/ languages (not core Algerian data) ---
    aux_counts = {}
    for lang_dir in sorted((DATA_DIR / "sentiment").iterdir()):
        if not lang_dir.is_dir() or lang_dir.name in ("ar_dz", "ar_na"):
            continue
        counts = {}
        for split in ("train", "dev", "test"):
            path = lang_dir / f"{split}.csv"
            if path.exists():
                counts[split] = sum(1 for _ in path.open("r", encoding="utf-8", errors="replace"))
        aux_counts[lang_dir.name] = counts
    report["auxiliary_sentiment_languages_row_counts"] = aux_counts

    print(json.dumps(report, ensure_ascii=False, indent=2))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "narabizi_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
