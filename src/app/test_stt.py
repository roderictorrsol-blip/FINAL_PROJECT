from src.app.voice_utils import transcribe_audio


if __name__ == "__main__":
    audio_path = "data/test_audio.wav"
    text = transcribe_audio(audio_path)
    print(text)