"""
01_gather_all.py

Batch-fetch YouTube transcripts for all video IDs in config.py and store them as JSON.
Also produces a merged JSON file with all successful transcripts.

Outputs:
- data/raw/t_01.json
- data/raw/t_02.json
- ...
- data/raw/all_transcripts.json
"""

# Enable postponed evaluation of type annotations.
from __future__ import annotations

# Standard library imports.
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Third-party imports.
import requests
from youtube_transcript_api import YouTubeTranscriptApi

# Project configuration.
from config import VIDEO_IDS, RAW_DIR, PREFERRED_LANGUAGES


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def project_root() -> Path:
    """
    Return the project root folder.

    This assumes the file lives under:
        project_root/src/01_gather_all.py
    """
    return Path(__file__).resolve().parents[1]


def t_raw_filename(index: int) -> str:
    """
    Return the standardized raw transcript filename.

    Example:
        1 -> t_01.json
        2 -> t_02.json
    """
    return f"t_{index:02d}.json"


def snippet_to_dict(snippet: Any) -> Dict[str, float | str]:
    """
    Normalize transcript snippets to a consistent schema.

    The YouTube transcript API may return either dictionaries or objects,
    so this helper converts both cases into a uniform dictionary with:
    - text
    - start
    - duration
    """
    # Handle dictionary-style transcript items.
    if isinstance(snippet, dict):
        return {
            "text": (snippet.get("text") or "").replace("\n", " ").strip(),
            "start": float(snippet.get("start", 0.0)),
            "duration": float(snippet.get("duration", 0.0)),
        }

    # Handle object-style transcript items.
    return {
        "text": (getattr(snippet, "text", "") or "").replace("\n", " ").strip(),
        "start": float(getattr(snippet, "start", 0.0)),
        "duration": float(getattr(snippet, "duration", 0.0)),
    }


def fetch_segments_with_retries(
    api: YouTubeTranscriptApi,
    video_id: str,
    max_attempts: int = 3,
    sleep_seconds: float = 2.0,
) -> Tuple[List[Dict[str, float | str]], str]:
    """
    Fetch transcript segments for a single video with retries.

    Strategy:
    - Try the preferred languages first
    - If that fails, fall back to any available transcript
    - Retry network and other transient errors a few times

    Returns
    -------
    Tuple[List[Dict[str, float | str]], str]
        A tuple containing:
        - the normalized transcript segments
        - the detected language code
    """

    # Keep track of the last error so it can be raised after all retries fail.
    last_err: Exception | None = None

    # Retry the fetch operation a limited number of times.
    for attempt in range(1, max_attempts + 1):
        try:
            # First try the preferred transcript languages from config.py.
            try:
                transcript = api.fetch(video_id, languages=PREFERRED_LANGUAGES)
            except Exception:
                # Fall back to any available transcript language.
                transcript = api.fetch(video_id)

            # Some versions may expose snippets as an attribute; others may be iterable directly.
            snippets = getattr(transcript, "snippets", transcript)

            # Normalize each transcript snippet into the project's standard schema.
            segments = [snippet_to_dict(s) for s in snippets]

            # Drop empty transcript lines.
            segments = [s for s in segments if str(s.get("text", "")).strip()]

            # Read the transcript language code when available.
            lang = getattr(transcript, "language_code", "unknown")

            return segments, lang

        except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
            # Network-related errors are retried with incremental backoff.
            last_err = e
            print(f"   WARN (network) attempt {attempt}/{max_attempts}: {type(e).__name__}: {e}")
            time.sleep(sleep_seconds * attempt)
            continue

        except Exception as e:
            # Other errors are also retried because some API failures may be temporary.
            last_err = e
            print(f"   WARN attempt {attempt}/{max_attempts}: {type(e).__name__}: {e}")
            time.sleep(sleep_seconds * attempt)
            continue

    # If all attempts fail, raise the last captured exception.
    assert last_err is not None
    raise last_err


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    """
    Save a JSON payload to disk using UTF-8 and pretty formatting.
    """
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    """
    Fetch transcripts for all configured YouTube videos and save them to disk.

    This script:
    - creates the raw output directory if needed
    - downloads one transcript per configured video
    - stores each transcript individually
    - maintains a merged JSON file with partial progress
    """
    # Resolve and create the output directory for raw transcript files.
    out_dir = project_root() / RAW_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create the YouTube transcript API client.
    api = YouTubeTranscriptApi()

    # Track total configured videos and number of successful downloads.
    total = len(VIDEO_IDS)
    ok = 0

    # Accumulate successful transcript payloads for the merged JSON output.
    merged: List[dict] = []

    # Path of the merged transcript file.
    merged_path = out_dir / "all_transcripts.json"

    # Process each configured video one by one.
    for i, video_id in enumerate(VIDEO_IDS, start=1):
        # Build the internal transcript ID (t_01, t_02, ...).
        t_id = f"t_{i:02d}"

        # Build the expected per-video output path.
        out_file = out_dir / t_raw_filename(i)

        print(f"[{i:02d}/{total}] Fetching transcript for video_id={video_id}")

        # If the transcript was already downloaded, reuse it.
        if out_file.exists():
            try:
                # Load the existing file and include it in the merged output.
                existing = json.loads(out_file.read_text(encoding="utf-8"))
                merged.append(existing)
                ok += 1
                print(f"   SKIP -> already exists: {out_file}")
                continue
            except Exception:
                # If the existing file is corrupted or unreadable, fetch it again.
                print(f"   WARN -> existing file unreadable, re-fetching: {out_file}")

        try:
            # Download transcript segments with retry logic.
            segments, lang = fetch_segments_with_retries(api, video_id)

            # Build the normalized transcript payload for this video.
            payload = {
                "t_id": t_id,
                "video_id": video_id,
                "language": lang,
                "segments": segments,
            }

            # Save the individual transcript file.
            save_json(out_file, payload)

            # Add the successful payload to the merged output.
            merged.append(payload)
            ok += 1

            # Save partial merged progress after each success so work is never lost.
            save_json(merged_path, {"total_videos": ok, "videos": merged})

            print(f"   OK -> {out_file} | segments: {len(segments)} | lang: {lang}")

        except KeyboardInterrupt:
            # Save partial progress if the user interrupts execution.
            print("\nInterrupted by user. Saving partial merged JSON and exiting...")
            save_json(merged_path, {"total_videos": ok, "videos": merged})
            raise

        except Exception as e:
            # Log the error and continue with the remaining videos.
            print(f"   ERROR -> {type(e).__name__}: {e}")
            continue

    # Save the final merged transcript file after processing all videos.
    save_json(merged_path, {"total_videos": ok, "videos": merged})

    print(f"\nMerged JSON saved -> {merged_path}")
    print(f"Done. Success: {ok}/{total}. Output folder: {out_dir}")


# Run the script only when executed directly.
if __name__ == "__main__":
    main()