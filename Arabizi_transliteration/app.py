import gradio as gr
from transliterate import transliterate

description = (
    "This tool transliterates Algerian Darija text from Arabic script to Arabizi (Latin script). "
    "Because Arabizi has no standard orthography and speakers write the same words in multiple ways, "
    "the output is deliberately non-deterministic. Submit the input multiple times or try different "
    "runs to explore different valid spelling variations for the same text."
)

examples = [
    ["راني عارف"],
    ["عليهم"],
    ["نصرو شاك داير فيها"],
    ["ربي يحفظك خويا"],
    ["الدار في وهران غير واه وشتاكاين"]
]

demo = gr.Interface(
    fn=transliterate,
    inputs=gr.Textbox(
        label="Input Arabic Script (Algerian Darija)",
        placeholder="Type here (e.g., راني عارف)...",
        lines=4
    ),
    outputs=gr.Textbox(
        label="Transliterated Arabizi",
        lines=4
    ),
    title="Algerian Darija Stochastic Arabizi Transliterator",
    description=description,
    examples=examples,
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Soft(
            primary_hue="amber",
            secondary_hue="stone",
        ),
    )
