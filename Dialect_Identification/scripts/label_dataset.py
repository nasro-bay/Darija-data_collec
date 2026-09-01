#!/usr/bin/env python
"""Labels data/unlabeled_10k.jsonl with Qwen3.5-4B (local, 4-bit -- see
../../local_models/) for the dialect/language ID task, per ../plan.md.

The deterministic script check (script_of(), same technique validated in
notebooks/01_cleaning_rules.ipynb) now drives BOTH which class the model
is even allowed to pick AND whether it's called at all -- not just a
rule it's told to follow:

- script == "mixed" (both Arabic and Latin script present) -> code_switch
  directly, model never called. Sidesteps a real, repeatedly-observed
  failure mode: the model reliably missed short embedded French/English
  phrases mid-sentence across two rounds of prompt tuning.
- script == "latin" (no Arabic script at all) -> model picks only from
  {arabize, french, english} via LATIN_PROMPT_TEMPLATE. msa/
  darija are structurally impossible to output, not just
  rule-forbidden -- a review batch found the model assigning
  darija to pure-Latin text at a 32% rate within that
  subgroup despite an explicit rule forbidding it and matching few-shot
  examples, so restricting the actual candidate set (rather than a
  fourth round of prompt tuning, or the post-hoc correct_label() this
  replaces) is what actually closes it.
- script == "arabic" (no Latin script at all) -> model picks only from
  {msa, darija} via ARABIC_PROMPT_TEMPLATE -- same
  structural guarantee in the other direction.
- script == "other" (emoji/digits/punctuation only, no real script
  content) -> not labeled at all (label recorded as "other", no model
  call) -- there's no linguistic signal to classify.

No confidence field (dropped per instruction).

One example at a time (not batched). Runs to completion over the whole
dataset by default -- the per-100-row manual review checkpoint used
during early tuning (see data/review_flags.jsonl for what it caught) is
now dropped; prompt/structural fixes derived from those reviews are
already baked into the two prompts below. Still resumable via
data/labeled_10k.jsonl's line count: if the process is interrupted
(killed, quota/power loss, etc.), rerunning this script picks up exactly
where it left off rather than relabeling from scratch.

Run via the GPU venv's Python:

    ".../ai-gpu/Scripts/python.exe" label_dataset.py
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
from pathlib import Path

# Model files are already fully cached locally from earlier runs -- force
# offline mode so from_pretrained() never attempts a Hub network call at
# all. Without this, an internal huggingface_hub/httpx client-lifecycle
# bug ("RuntimeError: Cannot send a request, as the client has been
# closed") can crash the run on an otherwise-unnecessary connectivity
# check; must be set before importing transformers/huggingface_hub.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Duplicated from build_dataset.py rather than imported -- that module
# needs datasketch (base env only), while this script needs the GPU
# venv's torch/transformers/bitsandbytes; keeping this function
# standalone avoids requiring both dependency sets in one environment.
# Derived and validated in notebooks/01_cleaning_rules.ipynb.
_MENTION_WITH_FRAGMENT_RE = re.compile(r"\[MENTION\](\s*-[^\s]{1,10})?")
_URL_RE = re.compile(r"\[URL\]")
_INLINE_WHITESPACE_RE = re.compile(r"[ \t]+")
_ARABIC_RE = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")
_LATIN_RE = re.compile(r"[a-zA-ZÀ-ɏ]")


def clean_for_classification(text: str) -> str:
    text = _MENTION_WITH_FRAGMENT_RE.sub("", text)
    text = _URL_RE.sub("", text)
    return _INLINE_WHITESPACE_RE.sub(" ", text).strip()


def script_of(text: str) -> str:
    has_ar = bool(_ARABIC_RE.search(text))
    has_lat = bool(_LATIN_RE.search(text))
    if has_ar and has_lat:
        return "mixed"
    if has_ar:
        return "arabic"
    if has_lat:
        return "latin"
    return "other"


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
UNLABELED_PATH = DATA_DIR / "unlabeled_10k.jsonl"
LABELED_PATH = DATA_DIR / "labeled_10k.jsonl"

MODEL_ID = "Qwen/Qwen3.5-4B"

# Self-attention memory scales roughly with sequence length squared -- most
# rows are short (median 49 chars across the 20k pool), but a rare long tail
# of djelfa forum posts runs into the tens of thousands of characters (max
# observed: 42,564). One such post (~23k chars) blew a single generate()
# call's attention allocation past the 6GB card's budget on top of the
# ~3.3GB the 4-bit model itself holds. 500 chars sits just above the p90
# (230) -- only the ~4% long-tail rows are affected, and language/script/
# dialect identification doesn't need the whole post, just a representative
# sample of it. Truncated at PROMPT time only (transient, same "clean at
# point of use, keep raw text stored" convention as clean_for_classification)
# -- the stored/labeled row always keeps the full original text.
MAX_PROMPT_CHARS = 500

LATIN_CLASSES = ["arabize", "french", "english"]
ARABIC_CLASSES = ["msa", "darija"]

LATIN_PROMPT_TEMPLATE = """You are a language identification classifier for Algerian online text.
This text is already known to be written entirely in Latin/ASCII script.

Your task is to classify it into **exactly one** of these three classes:

* `arabize`
* `french`
* `english`

### Definitions

* **arabize**: Arabic/Darija written entirely in Latin/ASCII characters (Arabizi), such as `wach`, `3lach`, `kifach`, `machi`, `saha`, `rani`, `7`, `3`, `9`, etc.
* **french**: Text primarily written in French.
* **english**: Text primarily written in English.

### Rules

1. Ignore emojis, usernames, and numbers when judging the language.
2. Read the entire text, not just the start, before deciding.
3. Return your answer in **exactly** this format, nothing else -- no explanation:

label: <one of arabize, french, english>

### Examples

Text: `wach rak kho`
label: arabize

Text: `Rani nseftar daba`
label: arabize

Text: `wallah 3endi 7chouma bezaf`
label: arabize

Text: `Bonjour, comment allez-vous ?`
label: french

Text: `Je pense que cette décision est une bonne idée pour l'avenir du pays.`
label: french

Text: `This is a very good video`
label: english

Text: `Thank you so much for sharing this information with us.`
label: english

### Input

Classify this text:

{text}
"""

ARABIC_PROMPT_TEMPLATE = """You are a language identification classifier for Algerian online text.
This text is already known to be written entirely in Arabic script.

Your task is to classify it into **exactly one** of these two classes:

* `msa`
* `darija`

### Definitions

* **msa**: Modern Standard Arabic / formal Arabic, including formal news, religious, governmental, or standard written Arabic.
* **darija**: Algerian Darija written using Arabic script. Look for Algerian dialect features such as `وعلاه`, `ماشي`, `هاذ`, `واش`, `راهو`, `نحب`, `بزاف`, `الزواولة`, etc. A single French/English loanword (e.g. `portable`, `ok`) does not change this label.

### Rules

1. Ignore emojis, usernames, numbers, and the placeholders `[MENTION]`/`[URL]` when judging the language.
2. Read the entire text, not just the start, before deciding.
3. Return your answer in **exactly** this format, nothing else -- no explanation:

label: <one of msa, darija>

### Examples

Text: `وعلاه تجيبهم أخر دقيقة`
label: darija

Text: `صحيت خويا الفيديو رائع بصح جيب لينا حاجة على portable الجديد`
label: darija

Text: `عيد الأضحى يوم 26 ماي هو غدا`
label: msa

Text: `أعلنت الحكومة الجزائرية عن خطة جديدة لدعم الفلاحين في المناطق الريفية`
label: msa

### Input

Classify this text:

{text}
"""


def load_unlabeled() -> list[dict]:
    with UNLABELED_PATH.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def already_labeled_count() -> int:
    if not LABELED_PATH.exists():
        return 0
    with LABELED_PATH.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def parse_response(raw: str, allowed_classes: list[str]) -> str:
    """Returns the label; 'parse_error' if the model's output doesn't
    contain a 'label:' line naming a class from `allowed_classes` --
    surfaced for review like anything else, not silently discarded or
    retried. `allowed_classes` is the structurally-restricted set for
    whichever prompt was used (see module docstring) -- validating
    against it here, not the full 5-class list, means a model output
    that violates the restriction (e.g. answers "msa" when given the
    Latin-only prompt) is caught as parse_error rather than silently
    accepted as a script-impossible label.
    """
    label = None
    for line in raw.strip().splitlines():
        line = line.strip().lower()
        if line.startswith("label:"):
            label = line.split(":", 1)[1].strip()
            break
    if label not in allowed_classes:
        return "parse_error"
    return label


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="rows to label this run (default: all remaining rows, i.e. run to completion); "
        "pass a small number for a quick smoke test",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available in this Python environment. Run this script via the "
            "GPU venv's interpreter -- the base environment's torch build is CPU-only."
        )

    rows = load_unlabeled()
    start = already_labeled_count()
    if start >= len(rows):
        print(f"All {len(rows):,} rows already labeled -- nothing to do.")
        return

    end = len(rows) if args.limit is None else min(start + args.limit, len(rows))
    print(f"Resuming at row {start:,}/{len(rows):,} -- labeling rows {start:,}..{end - 1:,} this run.")

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("Loading Qwen3.5-4B in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="auto",
    )
    print(f"Model loaded. VRAM allocated: {torch.cuda.memory_allocated() / 1e6:.0f} MB")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    label_counts: dict[str, int] = {}
    with LABELED_PATH.open("a", encoding="utf-8") as out:
        for i in range(start, end):
            row = rows[i]
            cleaned = clean_for_classification(row["text"])
            script = script_of(cleaned)
            truncated = len(cleaned) > MAX_PROMPT_CHARS

            if script == "mixed":
                # Regex decides code_switch directly -- model never called
                # for this row.
                label = "code_switch"
                raw = None
            elif script == "other":
                # No real script content (emoji/digits/punctuation only) --
                # no signal to classify, not labeled at all.
                label = "other"
                raw = None
            else:
                if script == "latin":
                    prompt_template, allowed = LATIN_PROMPT_TEMPLATE, LATIN_CLASSES
                else:  # "arabic"
                    prompt_template, allowed = ARABIC_PROMPT_TEMPLATE, ARABIC_CLASSES

                prompt = prompt_template.format(text=cleaned[:MAX_PROMPT_CHARS])
                messages = [{"role": "user", "content": prompt}]

                inputs = processor.apply_chat_template(
                    messages, tokenize=True, return_dict=True, return_tensors="pt",
                    add_generation_prompt=True, enable_thinking=False,
                ).to(model.device)
                output = model.generate(
                    **inputs, max_new_tokens=20, do_sample=False,
                )
                raw = processor.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
                label = parse_response(raw, allowed)

            label_counts[label] = label_counts.get(label, 0) + 1

            record = {
                "id": row["id"],
                "text": row["text"],
                "sample_group": row["sample_group"],
                "label": label,
                "raw_model_output": raw.strip() if label == "parse_error" else None,
                "truncated": truncated,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            os.fsync(out.fileno())

            if (i - start + 1) % 100 == 0:
                print(f"  ...{i - start + 1}/{end - start} labeled this run ({i + 1}/{len(rows)} overall)")

    print(f"\nDone. Labeled {end - start} rows this run ({end:,}/{len(rows):,} overall).")
    print("Label counts this run:", label_counts)
    if end < len(rows):
        print("(Interrupted or --limit given -- rerun this script to resume from here.)")


if __name__ == "__main__":
    main()
