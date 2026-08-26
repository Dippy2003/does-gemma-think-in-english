import gradio as gr

with gr.Blocks(title="Does Gemma think in English?") as demo:
    gr.Markdown("# Does a multilingual LLM pivot through English when prompted in Sinhala?")

if __name__ == "__main__":
    demo.launch()
