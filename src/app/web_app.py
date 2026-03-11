"""
app.py

Gradio web application for the WWII RAG assistant.

Responsibilities:
- Initialize the RAG pipeline lazily
- Receive text or voice input from the user
- Generate grounded answers using the RAG pipeline
- Optionally synthesize the answer as speech
- Display cited video sources in the UI
"""

# Enable postponed evaluation of type annotations.
from __future__ import annotations

# Standard library imports.
import logging
import os

# Third-party imports.
import gradio as gr

# Project imports.
from src.app.voice_utils import synthesize_speech, transcribe_audio
from src.pipeline.rag_pipeline import RAGPipeline


# Configure application-wide logging.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global pipeline instance reused across requests.
pipeline = None


def get_pipeline():
    """
    Lazily initialize and return the shared RAG pipeline instance.

    The pipeline is created only once and then reused for subsequent requests,
    which reduces startup overhead during the chat session.
    """
    global pipeline

    if pipeline is None:
        logger.info("Initializing RAG pipeline...")
        pipeline = RAGPipeline()

    return pipeline


def format_sources(docs):
    """
    Format retrieved source documents into a Markdown block for the UI.

    Each source includes:
    - video title
    - timestamped link
    - optional thumbnail preview

    The function also removes duplicates and limits the number of displayed sources.
    """
    # Return an empty string if no documents are available.
    if not docs:
        return ""

    # Start the sources section with a Markdown heading.
    lines = ["", "---", "### 📚 Fuentes consultadas"]

    # Track already displayed sources to avoid duplicates.
    seen = set()

    # Count displayed sources and limit them for readability.
    count = 0
    max_display_sources = 5

    for doc in docs:
        # Read document metadata safely.
        meta = getattr(doc, "metadata", {}) or {}

        # Prefer timestamped URLs when available.
        url = meta.get("source_url_t") or meta.get("source_url")

        # Read supporting metadata with fallbacks.
        timestamp = meta.get("start_hhmmss") or "tiempo desconocido"
        video_title = meta.get("video_title") or meta.get("video_id", "video")
        thumbnail_url = meta.get("thumbnail_url")

        # Use URL + timestamp as the deduplication key.
        key = (url, timestamp)

        # Skip invalid or repeated sources.
        if not url or key in seen:
            continue

        seen.add(key)
        count += 1

        # Stop when the maximum number of displayed sources is reached.
        if count > max_display_sources:
            break

        # If a thumbnail is available, render a richer source card.
        if thumbnail_url:
            lines.append(
                f"""
**{count}. {video_title}**

[![Miniatura del vídeo]({thumbnail_url})]({url})

[▶ Ver fragmento ({timestamp})]({url})
"""
            )
        else:
            # Fallback to a simpler source line without thumbnail.
            lines.append(
                f"**{count}. {video_title}** — [▶ Ver fragmento ({timestamp})]({url})"
            )

    # Return the formatted Markdown block only if at least one source was added.
    return "\n\n".join(lines) if count > 0 else ""


def respond(message, history, generate_tts):
    """
    Handle a text user message and update the chat history.

    Flow:
    1. Clean the input message
    2. Run the RAG pipeline
    3. Format the answer and cited sources
    4. Optionally generate TTS audio
    5. Return updated UI state
    """
    # Normalize the input message.
    clean_message = (message or "").strip()

    # Ignore empty inputs.
    if not clean_message:
        return "", history, None

    try:
        # Execute the RAG pipeline for the current user message.
        result = get_pipeline().run(clean_message)

        # Read the answer and supporting documents from the pipeline result.
        answer = result.get("answer", "No pude generar una respuesta.")
        docs = result.get("docs", [])

        # Append the formatted sources to the visible assistant answer.
        final_answer = f"{answer}{format_sources(docs)}"

        # Update the conversation history shown in the chat UI.
        history = history + [
            {"role": "user", "content": clean_message},
            {"role": "assistant", "content": final_answer},
        ]

        # Optionally generate audio only for the plain answer, not the sources block.
        audio_file = text_to_speech(answer) if generate_tts else None

        return "", history, audio_file

    except Exception as e:
        # Log the full traceback for debugging.
        logger.exception("Chat error")

        # Add a visible error message to the conversation history.
        history = history + [
            {"role": "user", "content": clean_message},
            {
                "role": "assistant",
                "content": f"Ha ocurrido un error al procesar la consulta: {e}",
            },
        ]

        return "", history, None


def audio_to_text(audio_path):
    """
    Convert a recorded audio question into text using speech-to-text.
    """
    # Return an empty string if no audio file was provided.
    if audio_path is None:
        return ""

    try:
        # Transcribe the input audio file into text.
        text = transcribe_audio(audio_path)
        return text
    except Exception:
        # Log transcription errors and fail gracefully.
        logger.exception("STT error")
        return ""


def text_to_speech(answer: str):
    """
    Convert the assistant's answer into speech and return the audio file path.
    """
    try:
        # Generate the audio response file for the assistant answer.
        audio_path = synthesize_speech(answer, "data/tts_response.mp3")
        return audio_path
    except Exception:
        # Log synthesis errors and fail gracefully.
        logger.exception("TTS error")
        return None


# Custom CSS used to improve the visual presentation of the Gradio app.
custom_css = """
footer {visibility: hidden;}

.gradio-container {
    max-width: 980px !important;
    margin: 0 auto !important;
}

#app-title {
    text-align: center;
    margin-bottom: 0.35rem;
}

#app-subtitle {
    text-align: center;
    color: #666;
    margin-bottom: 1.2rem;
}

#helper-text {
    text-align: center;
    font-size: 0.95rem;
    color: #777;
    margin-bottom: 1rem;
}

#intro-box {
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 14px 18px;
    margin-bottom: 18px;
    background: #fafafa;
}
"""


# Build the Gradio interface.
with gr.Blocks(
    title="Asistente sobre la Segunda Guerra Mundial",
) as demo:
    # Main title and user-facing explanation.
    gr.Markdown(
        """
        <h1 id="app-title">🌍 Explora la Segunda Guerra Mundial con IA</h1>
        <p id="app-subtitle">
            Consulta cualquier tema sobre la Segunda Guerra Mundial y recibe respuestas basadas en contenido real de videos documentales.
        </p>
        <p id="helper-text">
            Cada respuesta incluye fragmentos de videos utilizados como fuente.
        </p>
        """
    )

    # Introductory box describing what the assistant can help with.
    gr.Markdown(
        """
        <div id="intro-box">
            <strong>Explora temas como:</strong> orígenes de la guerra, operaciones militares, líderes mundiales, fechas clave y consecuencias históricas.
        </div>
        """
    )

    with gr.Group():
        # Main chat component with the initial assistant welcome message.
        chatbot = gr.Chatbot(
            value=[
                {
                    "role": "assistant",
                    "content": "¡Hola! Puedo ayudarte a responder preguntas sobre la Segunda Guerra Mundial a partir de transcripciones de vídeos.\n\nPrueba con una pregunta como: **¿Qué ocurrió en el Día D?**",
                }
            ],
            height=520,
            label="Conversación",
        )

        # Audio output for optional text-to-speech responses.
        tts_audio = gr.Audio(
            label="Respuesta en audio",
            type="filepath",
            autoplay=True,
        )

        # Main text input for user questions.
        msg = gr.Textbox(
            placeholder="Haz una pregunta sobre la Segunda Guerra Mundial...",
            label="Tu pregunta",
            lines=2,
        )

        # Toggle that enables/disables audio generation for responses.
        generate_tts = gr.Checkbox(
            label="Generar respuesta en audio",
            value=False,
        )

        # Voice input component for spoken user questions.
        audio_input = gr.Audio(
            sources=["microphone"],
            type="filepath",
            label="Pregunta por voz",
        )

        with gr.Row():
            # Submit the question and clear the chat history.
            submit_btn = gr.Button("Enviar", variant="primary")
            clear_btn = gr.Button("Limpiar chat")

        # Automatically transcribe microphone input into the text box.
        audio_input.change(
            audio_to_text,
            inputs=audio_input,
            outputs=msg,
        )

    # Example questions to help users explore the assistant.
    gr.Markdown("### Ejemplos de preguntas")

    gr.Examples(
        examples=[
            ["¿Qué causó la Segunda Guerra Mundial?"],
            ["¿Qué ocurrió en el Día D?"],
            ["¿Qué países formaban las Potencias del Eje?"],
            ["¿Por qué Alemania invadió Polonia en 1939?"],
            ["¿Qué papel tuvo la Unión Soviética en la guerra?"],
            ["¿Qué fue la batalla de Stalingrado?"],
        ],
        inputs=msg,
    )

    # Footer description of the application.
    gr.Markdown(
        """
        ---
        Asistente histórico basado en transcripciones de vídeos y recuperación semántica (RAG).
        """
    )

    # Send the current question when the submit button is clicked.
    submit_btn.click(
        respond,
        inputs=[msg, chatbot, generate_tts],
        outputs=[msg, chatbot, tts_audio],
    )

    # Send the current question when Enter is pressed in the textbox.
    msg.submit(
        respond,
        inputs=[msg, chatbot, generate_tts],
        outputs=[msg, chatbot, tts_audio],
    )

    # Reset the text box, chat history, and audio output.
    clear_btn.click(lambda: ("", [], None), outputs=[msg, chatbot, tts_audio])


# Launch the Gradio app when this file is executed directly.
if __name__ == "__main__":
    # Read the port from the environment, falling back to the default Gradio port.
    port = int(os.getenv("PORT", "7860"))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=True,
        theme=gr.themes.Soft(),
        css=custom_css,
    )