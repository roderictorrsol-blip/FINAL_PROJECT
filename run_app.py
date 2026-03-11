"""
run_app.py

Convenience entrypoint for launching the Gradio application.

This allows running the project with:

    python run_app.py

instead of:

    python -m src.app.app
"""

from src.app.web_app import demo


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
    )