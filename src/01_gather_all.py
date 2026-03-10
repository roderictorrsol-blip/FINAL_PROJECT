"""
01_gather_all.py

Batch-fetch YouTube transcripts for all video IDs in config.py and store them as JSON.
Also produces a merged JSON with all successful transcripts.

Outputs:
- data/raw/t_01.json
- data/raw/t_02.json
- ...
- data/raw/all_transcripts.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests
from youtube_transcript_api import YouTubeTranscriptApi

from config import VIDEO_IDS, RAW_DIR, PREFERRED_LANGUAGES


# -------------------------
# Helpers
# -------------------------

def project_root() -> Path:
    """Return the project root folder (FINAL_PROJECT)."""
    return Path(__file__).resolve().parents[1]


def t_raw_filename(index: int) -> str:
    """Return standardized raw filename: t_01.json, t_02.json, ..."""
    return f"t_{index:02d}.json"


def snippet_to_dict(snippet: Any) -> Dict[str, float | str]:
    """Normalize transcript snippets to a consistent schema."""
    if isinstance(snippet, dict):
        return {
            "text": (snippet.get("text") or "").replace("\n", " ").strip(),
            "start": float(snippet.get("start", 0.0)),
            "duration": float(snippet.get("duration", 0.0)),
        }
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
    Returns (segments, language_code).
    """

    last_err: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            # Try preferred languages first, then fallback
            try:
                transcript = api.fetch(video_id, languages=PREFERRED_LANGUAGES)
            except Exception:
                transcript = api.fetch(video_id)

            snippets = getattr(transcript, "snippets", transcript)
            segments = [snippet_to_dict(s) for s in snippets]
            segments = [s for s in segments if str(s.get("text", "")).strip()]
            lang = getattr(transcript, "language_code", "unknown")
            return segments, lang

        except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
            last_err = e
            print(f"   WARN (network) attempt {attempt}/{max_attempts}: {type(e).__name__}: {e}")
            time.sleep(sleep_seconds * attempt)
            continue

        except Exception as e:
            # Other errors (disabled transcript, no transcript, etc.)
            last_err = e
            print(f"   WARN attempt {attempt}/{max_attempts}: {type(e).__name__}: {e}")
            time.sleep(sleep_seconds * attempt)
            continue

    assert last_err is not None
    raise last_err


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# -------------------------
# Main
# -------------------------

def main() -> None:
    out_dir = project_root() / RAW_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    api = YouTubeTranscriptApi()

    total = len(VIDEO_IDS)
    ok = 0
    merged: List[dict] = []

    merged_path = out_dir / "all_transcripts.json"

    for i, video_id in enumerate(VIDEO_IDS, start=1):
        t_id = f"t_{i:02d}"
        out_file = out_dir / t_raw_filename(i)

        print(f"[{i:02d}/{total}] Fetching transcript for video_id={video_id}")

        # If already downloaded, skip (useful after an interruption)
        if out_file.exists():
            try:
                existing = json.loads(out_file.read_text(encoding="utf-8"))
                merged.append(existing)
                ok += 1
                print(f"   SKIP -> already exists: {out_file}")
                continue
            except Exception:
                # If file is corrupted, re-fetch
                print(f"   WARN -> existing file unreadable, re-fetching: {out_file}")

        try:
            segments, lang = fetch_segments_with_retries(api, video_id)

            payload = {
                "t_id": t_id,
                "video_id": video_id,
                "language": lang,
                "segments": segments,
            }

            save_json(out_file, payload)
            merged.append(payload)
            ok += 1

            # Save partial merged output so you never lose progress
            save_json(merged_path, {"total_videos": ok, "videos": merged})

            print(f"   OK -> {out_file} | segments: {len(segments)} | lang: {lang}")

        except KeyboardInterrupt:
            print("\nInterrupted by user. Saving partial merged JSON and exiting...")
            save_json(merged_path, {"total_videos": ok, "videos": merged})
            raise

        except Exception as e:
            print(f"   ERROR -> {type(e).__name__}: {e}")
            # Continue processing the rest
            continue

    save_json(merged_path, {"total_videos": ok, "videos": merged})
    print(f"\nMerged JSON saved -> {merged_path}")
    print(f"Done. Success: {ok}/{total}. Output folder: {out_dir}")


if __name__ == "__main__":
    main()