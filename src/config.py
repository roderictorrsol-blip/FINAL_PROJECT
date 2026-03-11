"""
config.py

Central project configuration.

This module defines:
- The YouTube videos used as the dataset
- Output directories for raw transcripts and processed chunks
- Preferred transcript languages for ingestion
"""

from __future__ import annotations

# ------------------------------------------------------------------
# YouTube dataset
# ------------------------------------------------------------------
# These video IDs define the corpus used for the RAG knowledge base.
# Each video will be downloaded, transcribed, and converted into chunks.

VIDEO_IDS: list[str] = [
    # t_01
    "fIRM0bEZTpc",

    # t_02 ... t_08
    "MBtHiyY3FPo",
    "7F_HjxCLo6o",
    "JjSvBhc1gcg",
    "Qro3G8lKl0Y",
    "_nsLed3AHy8",
    "85UTQ6OWDFU",
    "KrHmB8nCU3U",
]


# ------------------------------------------------------------------
# Pipeline directories
# ------------------------------------------------------------------

# Directory where raw transcripts are stored after ingestion.
RAW_DIR: str = "data/raw"

# Directory where processed transcript chunks are stored.
# These chunks are later used to build the vector database.
CHUNKS_DIR: str = "data/chunks"


# ------------------------------------------------------------------
# Transcript language preference
# ------------------------------------------------------------------

# Preferred transcript languages when fetching subtitles from YouTube.
# The system will attempt Spanish first, and if unavailable,
# it will fallback to any available language.
PREFERRED_LANGUAGES: list[str] = ["es"]