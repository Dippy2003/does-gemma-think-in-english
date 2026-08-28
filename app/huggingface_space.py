"""HuggingFace Spaces entrypoint — Spaces runs whatever file is set as
`app_file` in the Space's README front matter; this just re-exports the
Gradio Blocks object from app.app so a Space pointed at this file works
identically to running `python app/app.py` locally."""

from app.app import demo

if __name__ == "__main__":
    demo.launch()
