from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
from openai import OpenAI


client = OpenAI()


def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe an audio file to text using OpenAI speech-to-text.
    """
    path = Path(audio_path)

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    with path.open("rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=audio_file,
        )

    return transcript.text.strip()


def synthesize_speech(text: str, output_path: str) -> str:
    """
    Convert text to speech and save it to an mp3 file.
    Returns the output file path.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text,
    ) as response:
        response.stream_to_file(path)

    return str(path)