"""Single-turn chat inference against Qwen3.5-4B in 4-bit. See plan.md and
load_model.py's docstring for why this uses AutoProcessor/
AutoModelForImageTextToText instead of a plain AutoTokenizer/
AutoModelForCausalLM (Qwen3.5-4B is multimodal). Run via the GPU venv's
Python:

    ".../ai-gpu/Scripts/python.exe" chat.py
"""
import io
import sys

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

# Windows' console defaults to a codepage (cp1252) that can't encode
# Darija's Arabic-script output -- reconfigure stdout to UTF-8 (same fix
# needed for chat.py/Atlas-Chat-2B).
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

MODEL_ID = "Qwen/Qwen3.5-4B"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA is not available in this Python environment. Run this script via the "
        "GPU venv's interpreter -- the base environment's torch build is CPU-only."
    )

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
)

messages = [
    {"role": "user", "content": "kidayr? chno khbarek lyoum?"}
]

inputs = processor.apply_chat_template(
    messages, tokenize=True, return_dict=True, return_tensors="pt",
    add_generation_prompt=True, enable_thinking=False,
).to(model.device)
# return_dict=True is required here regardless of transformers version --
# same BatchEncoding-vs-bare-tensor gotcha hit with Atlas-Chat's chat.py.
# enable_thinking=False -- per plan.md's own warning, Qwen 3-series
# defaults to a "thinking" mode that burns the token budget on a visible
# reasoning trace before ever reaching a final answer; confirmed live,
# 256 max_new_tokens ran out mid-thought with no real answer at all.

output = model.generate(
    **inputs,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
)

response = processor.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
print(response)
