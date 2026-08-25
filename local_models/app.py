#!/usr/bin/env python
"""Gradio chat UI for Qwen3.5-4B (4-bit, see load_model.py/chat.py) --
same Algerian identity/styling as Embeddings/word2vec/app.py (itself
matching Embeddings/intrinsic_eval's Flask app): green/red/gold on cream,
Cairo font, DarijaDZ logo + flag strip, no gradients, SVG icons not emoji.

Qwen3.5-4B is multimodal (see load_model.py's docstring), hence
AutoProcessor/AutoModelForImageTextToText instead of a plain
AutoTokenizer/AutoModelForCausalLM.

Run via the GPU venv's Python (see requirements.txt):

    ".../ai-gpu/Scripts/python.exe" app.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import gradio as gr
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

# Windows' console defaults to a codepage that can't encode Darija's
# Arabic-script output -- reconfigure stdout to UTF-8 (same fix as chat.py).
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

MODEL_ID = "Qwen/Qwen3.5-4B"
STATIC_DIR = Path(__file__).resolve().parent / "static"

if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA is not available in this Python environment. Run this script via the "
        "GPU venv's interpreter (see requirements.txt) -- the base environment's "
        "torch build is CPU-only."
    )

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
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
)
vram_mb = torch.cuda.memory_allocated() / 1e6
print(f"Model loaded. VRAM allocated: {vram_mb:.0f} MB")


def generate_reply(messages: list[dict]) -> str:
    # Gradio runs this callback in a worker thread (anyio.to_thread), not
    # the main thread that loaded the model onto the GPU -- explicitly
    # setting the CUDA device here guards against a "CUDA error: unknown
    # error" that can surface when a fresh thread's CUDA context isn't
    # properly associated with the active device yet.
    torch.cuda.set_device(0)
    # enable_thinking=False -- Qwen 3-series defaults to a "thinking" mode
    # that burns the token budget on a visible reasoning trace before ever
    # reaching a final answer (see chat.py's comment; confirmed live).
    inputs = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt",
        add_generation_prompt=True, enable_thinking=False,
    ).to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    return processor.decode(output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)


def respond(user_message: str, display_history: list[dict], messages: list[dict]):
    if not user_message or not user_message.strip():
        return display_history, messages, ""

    messages = messages + [{"role": "user", "content": user_message}]
    reply = generate_reply(messages)
    messages = messages + [{"role": "assistant", "content": reply}]

    display_history = display_history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": reply},
    ]
    return display_history, messages, ""


def clear_chat():
    return [], [], ""


CUSTOM_CSS = """
:root {
  --dz-green: #04663a;
  --dz-red: #c8102e;
  --dz-white: #ffffff;
  --dz-gold: #b8922f;
  --dz-cream: #f7f4ee;
  --ink: #1e2723;
  --muted: #74827b;
  --border: #e2e0d8;
}

.gradio-container {
  font-family: "Cairo", "Segoe UI", Arial, sans-serif !important;
  background: var(--dz-cream) !important;
}

#topbar {
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 20px 28px;
  background: var(--dz-white);
  border: 1px solid var(--border);
  border-radius: 4px;
  margin-bottom: 4px;
}
#topbar img.logo {
  width: 44px;
  height: 44px;
  border-radius: 50%;
  border: 1px solid var(--border);
  object-fit: cover;
}
#topbar h1 {
  margin: 0;
  font-size: 1.3rem;
  font-weight: 700;
  color: var(--ink);
}
#topbar .subtitle {
  margin: 2px 0 0;
  font-size: 0.85rem;
  color: var(--muted);
}
#topbar .flag-mark {
  margin-left: auto;
  color: var(--dz-green);
}
#flagstrip {
  height: 4px;
  display: flex;
  margin-bottom: 22px;
  border-radius: 2px;
  overflow: hidden;
}
#flagstrip span { flex: 1; display: block; }
#flagstrip span:nth-child(1) { background: var(--dz-green); }
#flagstrip span:nth-child(2) { background: var(--dz-white); border-top: 1px solid var(--border); border-bottom: 1px solid var(--border); }
#flagstrip span:nth-child(3) { background: var(--dz-red); }

#chatbot {
  background: var(--dz-white) !important;
  border: 1px solid var(--border) !important;
  border-radius: 4px !important;
}
/* Gradio 6's Chatbot renders role classes as plain .user/.bot (wrapped in
   a zero-specificity :where() internally for its own scoped styles) --
   NOT ".message.user"/".message.bot", which is why an earlier version of
   this rule silently never matched anything and left the default
   soft-accent background with dark text (poor contrast against green). */
#chatbot .user {
  background: var(--dz-green) !important;
  color: white !important;
}
#chatbot .bot {
  background: var(--dz-cream) !important;
  color: var(--ink) !important;
}

#msg-row input[type="text"] {
  border-radius: 6px !important;
  border: 1px solid var(--border) !important;
  font-family: "Cairo", "Segoe UI", Arial, sans-serif !important;
  font-size: 1.05rem !important;
}
#send-btn {
  background: var(--dz-green) !important;
  color: white !important;
  border: none !important;
  border-radius: 6px !important;
  font-weight: 600 !important;
}
#send-btn:hover { background: #054d2c !important; }
#clear-btn {
  background: transparent !important;
  color: var(--muted) !important;
  border: 1px solid var(--border) !important;
  border-radius: 6px !important;
}
#clear-btn:hover { color: var(--dz-red) !important; border-color: var(--dz-red) !important; }
"""

TOPBAR_HTML = f"""
<div id="topbar">
  <img class="logo" src="/gradio_api/file={STATIC_DIR / 'logo.png'}" alt="DarijaDZ">
  <div>
    <h1>Darija Chat -- Qwen3.5-4B</h1>
    <p class="subtitle">4-bit local inference (RTX 2060, {vram_mb:.0f}MB VRAM)</p>
  </div>
  <div class="flag-mark" aria-hidden="true">
    <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor">
      <path d="M14.5 3.5a8.5 8.5 0 1 0 0 17 8.7 8.7 0 0 1 0-17z"/>
      <path d="m17.6 8.2.9 2.7h2.9l-2.3 1.7.9 2.7-2.4-1.7-2.3 1.7.9-2.7-2.4-1.7h2.9z"/>
    </svg>
  </div>
</div>
<div id="flagstrip"><span></span><span></span><span></span></div>
"""

with gr.Blocks(title="Darija Chat -- Qwen3.5-4B") as demo:
    messages_state = gr.State([])  # full role/content history sent to the model

    gr.HTML(TOPBAR_HTML)

    chatbot = gr.Chatbot(elem_id="chatbot", height=480, show_label=False)

    with gr.Row(elem_id="msg-row"):
        msg_box = gr.Textbox(
            placeholder="kidayr? chno khbarek lyoum?",
            show_label=False,
            scale=4,
        )
        send_btn = gr.Button("Send", elem_id="send-btn", scale=1)
        clear_btn = gr.Button("Clear", elem_id="clear-btn", scale=1)

    send_btn.click(
        fn=respond, inputs=[msg_box, chatbot, messages_state], outputs=[chatbot, messages_state, msg_box]
    )
    msg_box.submit(
        fn=respond, inputs=[msg_box, chatbot, messages_state], outputs=[chatbot, messages_state, msg_box]
    )
    clear_btn.click(fn=clear_chat, outputs=[chatbot, messages_state, msg_box])


if __name__ == "__main__":
    demo.launch(css=CUSTOM_CSS, allowed_paths=[str(STATIC_DIR)])
