"""
voice_utils.py

Voice utilities for the WWII RAG assistant.

Responsibilities:
- Convert microphone audio into text (speech-to-text)
- Convert assistant responses into speech (text-to-speech)

This module isolates all OpenAI audio functionality so the UI
layer can remain clean and focused on interaction logic.
"""

# Enable postponed evaluation of type annotations.
from __future__ import annotations

# Standard library imports.
import os
from pathlib import Path

# Third-party imports.
from dotenv import load_dotenv
from openai import OpenAI


# Load environment variables from the .env file.
# This allows the OpenAI API key to be accessed via environment variables.
load_dotenv()


def get_client() -> OpenAI:
    """
    Create and return an OpenAI client.

    The API key is read from environment variables (OPENAI_API_KEY).
    """
    api_key = os.getenv("OPENAI_API_KEY")

    # Ensure the API key is available before creating the client.
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set in the environment or .env file."
        )

    return OpenAI(api_key=api_key)


def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe an audio file into text using OpenAI speech-to-text.

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

    # Ensure the input audio file exists.
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Create OpenAI client.
    client = get_client()

    # Send the audio file to the OpenAI transcription API.
    with path.open("rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=audio_file,
        )

    # Return cleaned transcript text.
    return transcript.text.strip()


def synthesize_speech(text: str, output_path: str) -> str:
    """
    Convert text into speech using OpenAI TTS and save it to an MP3 file.

    Parameters
    ----------
    text : str
        The text to convert to speech.

    output_path : str
        Path where the generated audio file will be saved.

    Returns
    -------
    str
        Path to the generated audio file.
    """

    # Prevent empty or whitespace-only text from being synthesized.
    if not text or not text.strip():
        raise ValueError("Cannot synthesize speech from empty text.")

    path = Path(output_path)

    # Ensure the output directory exists.
    path.parent.mkdir(parents=True, exist_ok=True)

    # Create OpenAI client.
    client = get_client()

    # Generate speech and stream it directly to the output file.
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text.strip(),
    ) as response:
        response.stream_to_file(str(path))

    return str(path)