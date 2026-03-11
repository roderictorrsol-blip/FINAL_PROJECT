from __future__ import annotations

import logging
import os

import gradio as gr

from src.app.voice_utils import synthesize_speech, transcribe_audio
from src.pipeline.rag_pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pipeline = None


def get_pipeline():
    global pipeline
    if pipeline is None:
        logger.info("Initializing RAG pipeline...")
        pipeline = RAGPipeline()
    return pipeline


def format_sources(docs):
    if not docs:
        return ""

    lines = ["", "---", "### 📚 Fuentes consultadas"]
    seen = set()
    count = 0
    max_display_sources = 5

    for doc in docs:
        meta = getattr(doc, "metadata", {}) or {}

        url = meta.get("source_url_t") or meta.get("source_url")
        timestamp = meta.get("start_hhmmss") or "tiempo desconocido"
        video_title = meta.get("video_title") or meta.get("video_id", "video")
        thumbnail_url = meta.get("thumbnail_url")

        key = (url, timestamp)
        if not url or key in seen:
            continue

        seen.add(key)
        count += 1

        if count > max_display_sources:
            break

        if thumbnail_url:
            lines.append(
                f"""
**{count}. {video_title}**

[![Miniatura del vídeo]({thumbnail_url})]({url})

[▶ Ver fragmento ({timestamp})]({url})
"""
            )
        else:
            lines.append(
                f"**{count}. {video_title}** — [▶ Ver fragmento ({timestamp})]({url})"
            )

    return "\n\n".join(lines) if count > 0 else ""


def respond(message, history, generate_tts):
    clean_message = (message or "").strip()

    if not clean_message:
        return "", history, None

    try:
        result = get_pipeline().run(clean_message)

        answer = result.get("answer", "No pude generar una respuesta.")
        docs = result.get("docs", [])

        final_answer = f"{answer}{format_sources(docs)}"

        history = history + [
            {"role": "user", "content": clean_message},
            {"role": "assistant", "content": final_answer},
        ]

        audio_file = text_to_speech(answer) if generate_tts else None

        return "", history, audio_file

    except Exception as e:
        logger.exception("Chat error")
        history = history + [
            {"role": "user", "content": clean_message},
            {
                "role": "assistant",
                "content": f"Ha ocurrido un error al procesar la consulta: {e}",
            },
        ]
        return "", history, None


def audio_to_text(audio_path):
    if audio_path is None:
        return ""

    try:
        text = transcribe_audio(audio_path)
        return text
    except Exception:
        logger.exception("STT error")
        return ""


def text_to_speech(answer: str):
    try:
        audio_path = synthesize_speech(answer, "data/tts_response.mp3")
        return audio_path
    except Exception:
        logger.exception("TTS error")
        return None


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


with gr.Blocks(
    title="Asistente sobre la Segunda Guerra Mundial",
) as demo:
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

    gr.Markdown(
        """
        <div id="intro-box">
            <strong>Explora temas como:</strong> orígenes de la guerra, operaciones militares, líderes mundiales, fechas clave y consecuencias históricas.
        </div>
        """
    )

    with gr.Group():
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

        tts_audio = gr.Audio(
            label="Respuesta en audio",
            type="filepath",
            autoplay=True,
        )

        msg = gr.Textbox(
            placeholder="Haz una pregunta sobre la Segunda Guerra Mundial...",
            label="Tu pregunta",
            lines=2,
        )

        generate_tts = gr.Checkbox(
            label="Generar respuesta en audio",
            value=False,
        )

        audio_input = gr.Audio(
            sources=["microphone"],
            type="filepath",
            label="Pregunta por voz",
        )

        with gr.Row():
            submit_btn = gr.Button("Enviar", variant="primary")
            clear_btn = gr.Button("Limpiar chat")

        audio_input.change(
            audio_to_text,
            inputs=audio_input,
            outputs=msg,
        )

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

    gr.Markdown(
        """
        ---
        Asistente histórico basado en transcripciones de vídeos y recuperación semántica (RAG).
        """
    )

    submit_btn.click(
        respond,
        inputs=[msg, chatbot, generate_tts],
        outputs=[msg, chatbot, tts_audio],
    )

    msg.submit(
        respond,
        inputs=[msg, chatbot, generate_tts],
        outputs=[msg, chatbot, tts_audio],
    )

    clear_btn.click(lambda: ("", [], None), outputs=[msg, chatbot, tts_audio])


if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    is_cloud = "PORT" in os.environ

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=not is_cloud,
        theme=gr.themes.Soft(),
        css=custom_css,
    )