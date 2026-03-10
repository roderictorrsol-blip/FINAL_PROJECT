from __future__ import annotations

import uuid
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


def synthesize_speech(text: str, output_dir: str = "data") -> str:
    """
    Generate TTS audio with a unique filename and post-process it.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    uid = uuid.uuid4().hex

    raw_path = Path(output_dir) / f"tts_{uid}_raw.mp3"
    final_path = Path(output_dir) / f"tts_{uid}.mp3"

    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text,
    ) as response:
        response.stream_to_file(raw_path)

  
    return str(final_path)