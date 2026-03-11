from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env
load_dotenv()


def get_client() -> OpenAI:
    """
    Create and return an OpenAI client using the API key
    from environment variables.
    """
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set in the environment or .env file."
        )

    return OpenAI(api_key=api_key)


def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe an audio file to text using OpenAI speech-to-text.

    Parameters
    ----------
    audio_path : str
        Path to the audio file to be transcribed.

    Returns
    -------
    str
        The transcribed text.
    """
    path = Path(audio_path)

    # Ensure the audio file exists
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Create OpenAI client
    client = get_client()

    # Send the audio file to the OpenAI transcription API
    with path.open("rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=audio_file,
        )

    # Return cleaned transcript
    return transcript.text.strip()


def synthesize_speech(text: str, output_path: str) -> str:
    """
    Convert text to speech using OpenAI TTS and save the result to an MP3 file.

    Parameters
    ----------
    text : str
        The text to convert to speech.

    output_path : str
        Path where the generated audio file will be saved.

    Returns
    -------
    str
        The path of the generated audio file.
    """

    # Prevent empty text from being synthesized
    if not text or not text.strip():
        raise ValueError("Cannot synthesize speech from empty text.")

    path = Path(output_path)

    # Ensure the output directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Create OpenAI client
    client = get_client()

    # Generate speech and stream it to file
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text.strip(),
    ) as response:
        response.stream_to_file(str(path))

    return str(path)