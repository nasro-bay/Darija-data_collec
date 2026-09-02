#!/usr/bin/env python
"""One-time (well, one-*deliberate*-run) update: replaces the old
"Script Distribution" table (Arabic script / Latin / Digits / Mixed --
token-level, regex-only) in DarijaDZ/README.md and Kaggle_DarijaDz/
README.md with a "Dialect Distribution" table (msa/darija/arabize/
french/english/code_switch/other -- document-level), estimated by
classify_corpus.py's run of the project's best dialect-ID model over a
5,000,000-row sample of the real corpus.

Deliberately NOT wired into Youtube_scrap/scripts/build_unified_dataset.py
-- this table is a static snapshot from here on; see that script's module
docstring for why. Rerun this script by hand (after rerunning
classify_corpus.py) if the table ever needs a deliberate refresh.

Run via the base Python environment:

    python update_readme_tables.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DIST_PATH = ROOT / "Dialect_Identification" / "data" / "corpus_dialect_distribution.json"
README_PATHS = [ROOT / "DarijaDZ" / "README.md", ROOT / "Kaggle_DarijaDz" / "README.md"]

LABEL_ROWS = [
    ("darija", "Darija (Algerian dialect, Arabic script)"),
    ("msa", "MSA (Modern Standard Arabic)"),
    ("arabize", "Arabize (Darija in Latin script / Arabizi)"),
    ("code_switch", "Code-switched (Arabic + Latin script mixed)"),
    ("french", "French"),
    ("english", "English"),
    ("other", "Other (no linguistic content -- emoji, digits, symbols only)"),
]

# Matches either the original pre-migration heading ("Script Distribution",
# the very first run) or this section's own heading on a later deliberate
# refresh ("Dialect Distribution") -- either way, replaced with a freshly
# built section in the same (Documents-column-free, per the maintainers'
# own edit) format below.
OLD_SECTION_RE = re.compile(
    r"## (?:Script|Dialect) Distribution\n.*?(?=\n## |\Z)",
    re.DOTALL,
)


def build_section(dist: dict) -> str:
    pcts = dist["percentages"]
    sample_size = dist["sample_size"]

    rows = []
    for key, label in LABEL_ROWS:
        pct = pcts.get(key, 0.0)
        rows.append(f"| {label:<58} | **{pct:.2f}%** |")

    table = "\n".join(rows)
    return f"""## Dialect Distribution

The following statistics describe **dialect/language identity**, not writing script -- estimated by running this project's best-performing dialect-ID model (character n-gram TF-IDF + SVM-RBF, per `Dialect_Identification/notebooks/04_baldwin_lui_ngrams.ipynb`, 0.833/0.881 held-out test accuracy) over a random **{sample_size:,}-document sample** of the corpus. This is a **static snapshot**, not automatically recomputed on every corpus rebuild -- see `Dialect_Identification/scripts/best_model/`.

### Document-level distribution

| Category                                                    | Percentage |
| ------------------------------------------------------------ | ---------: |
{table}

The same imbalance between Arabic-script Darija and Arabizi (Darija in Latin script) motivated the development of an **Arabic-script → Arabizi transliteration tool** to help balance the data.

[darija-arabizi-transliterator](https://huggingface.co/spaces/nasrellahkharroubi/darija-arabizi-transliterator)
"""


def main() -> None:
    if not DIST_PATH.exists():
        raise SystemExit(f"{DIST_PATH} not found -- run classify_corpus.py first")
    dist = json.loads(DIST_PATH.read_text(encoding="utf-8"))
    new_section = build_section(dist).rstrip("\n") + "\n"

    for readme_path in README_PATHS:
        if not readme_path.exists():
            print(f"  WARNING: {readme_path} not found, skipping")
            continue
        text = readme_path.read_text(encoding="utf-8")
        new_text, n = OLD_SECTION_RE.subn(new_section, text, count=1)
        if n == 0:
            print(f"  WARNING: '## Script Distribution' section not found in {readme_path} -- left unchanged")
            continue
        readme_path.write_text(new_text, encoding="utf-8")
        print(f"  Updated {readme_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
