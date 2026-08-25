"""Smoke-test: loads Qwen3.5-4B in 4-bit (NF4, bitsandbytes) and reports
VRAM footprint. See plan.md.

Qwen3.5-4B is multimodal (Qwen3_5ForConditionalGeneration -- image+video+
text), unlike Atlas-Chat-2B's plain text-only CausalLM -- so this uses
AutoModelForImageTextToText/AutoProcessor instead of
AutoModelForCausalLM/AutoTokenizer. The exact repo id in plan.md
("Qwen/Qwen3.5-4B-Instruct") doesn't exist on HF; the real non-gated,
chat-template-equipped release is just "Qwen/Qwen3.5-4B" (no "-Instruct"
suffix -- it's already the finetuned/chat variant, distinct from
"Qwen/Qwen3.5-4B-Base").

Run via the GPU venv's Python:

    ".../ai-gpu/Scripts/python.exe" load_model.py
"""
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

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
print(f"Device: {torch.cuda.get_device_name(0)}")

processor = AutoProcessor.from_pretrained(MODEL_ID)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
)

vram_mb = torch.cuda.memory_allocated() / 1e6
print(f"Model loaded. VRAM allocated: {vram_mb:.0f} MB")
