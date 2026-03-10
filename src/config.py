"""
config.py

Central project configuration:
- YouTube video IDs to ingest
- Output directories for raw transcripts and chunked dataset
"""

from __future__ import annotations

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

RAW_DIR: str = "data/raw"
CHUNKS_DIR: str = "data/chunks"

# Try Spanish first; if it fails, fallback to "whatever is available"
PREFERRED_LANGUAGES: list[str] = ["es"]