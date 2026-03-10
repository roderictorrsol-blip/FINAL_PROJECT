from src.app.voice_utils import synthesize_speech

if __name__ == "__main__":
    text = "Esta es una prueba de voz del asistente histórico sobre la Segunda Guerra Mundial."

    path = synthesize_speech(text)

    print("Audio generado en:", path)