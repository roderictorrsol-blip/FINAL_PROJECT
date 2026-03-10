"""
02_chunk_all.py

Batch-chunk all raw transcript JSON files from data/raw/ into data/chunks/.
Also writes a merged dataset file for retrieval/indexing:

Outputs:
- data/chunks/t_01_chunks.json
- ...
- data/chunks/t_12_chunks.json
- data/chunks/all_chunks.json

Each chunk includes:
- chunk_id, video_id
- text
- start/end (seconds)
- start_hhmmss/end_hhmmss
- source_url (YouTube deep link with &t=...)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

from config import RAW_DIR, CHUNKS_DIR


# -------------------------
# Configuration
# -------------------------

MAX_CHARS: int = 300      # Approximate chunk size (characters)
OVERLAP_CHARS: int = 100   # Overlap to preserve context across chunks


def project_root() -> Path:
    """Return the project root folder (FINAL_PROJECT)."""
    return Path(__file__).resolve().parents[1]


def format_hhmmss(seconds: float) -> str:
    """Format seconds into mm:ss or hh:mm:ss."""
    sec = int(seconds)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def build_text_and_mapping(
    segments: List[Dict],
) -> Tuple[str, List[Tuple[int, int, float, float]]]:
    """
    Build a single concatenated transcript string and a mapping from
    character indices to timestamps.

    Returns:
        full_text: concatenated transcript text
        mapping: list of tuples (start_idx, end_idx, seg_start, seg_end)
    """
    pieces: List[str] = []
    mapping: List[Tuple[int, int, float, float]] = []
    idx = 0

    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue

        start = float(seg.get("start", 0.0))
        duration = float(seg.get("duration", 0.0))
        end = start + duration

        # Trailing space helps avoid word merges across segment boundaries.
        t = text.replace("\n", " ").strip() + " "

        i0 = idx
        i1 = idx + len(t)

        pieces.append(t)
        mapping.append((i0, i1, start, end))
        idx = i1

    return "".join(pieces).strip(), mapping


def span_to_time(
    a: int,
    b: int,
    mapping: List[Tuple[int, int, float, float]],
) -> Tuple[float, float]:
    """
    Convert a text character span [a, b) to a (start, end) timestamp interval
    based on overlapping transcript segments.
    """
    start_t = None
    end_t = None

    for (i0, i1, t0, t1) in mapping:
        if start_t is None and i1 > a:
            start_t = t0
        if i0 < b:
            end_t = t1

    if start_t is None:
        start_t = 0.0
    if end_t is None:
        end_t = start_t

    return float(start_t), float(end_t)


def chunk_transcript(video_id: str, segments: List[Dict]) -> List[Dict]:
    """
    Chunk one transcript into RAG-friendly passages.

    Each chunk is assigned:
    - timestamps derived from the character span
    - a source_url for direct YouTube linking at the chunk start
    """
    full_text, mapping = build_text_and_mapping(segments)
    n = len(full_text)

    chunks: List[Dict] = []
    i = 0

    while i < n:
        j = min(i + MAX_CHARS, n)

        # Prefer cutting at whitespace to avoid splitting words.
        if j < n:
            cut = full_text.rfind(" ", i, j)
            if cut != -1 and cut > i + 200:
                j = cut

        start_t, end_t = span_to_time(i, j, mapping)
        chunk_text = full_text[i:j].strip()

        chunks.append(
            {
                "chunk_id": f"{video_id}_{len(chunks):04d}",
                "video_id": video_id,
                "text": chunk_text,
                "start": start_t,
                "end": end_t,
                "start_hhmmss": format_hhmmss(start_t),
                "end_hhmmss": format_hhmmss(end_t),
                "source_url": f"https://www.youtube.com/watch?v={video_id}&t={int(start_t)}s",
            }
        )

        # Advance with overlap.
        next_i = j - OVERLAP_CHARS
        i = next_i if next_i > i else j
        if i <= 0:
            i = j

    return chunks


def main() -> None:
    raw_dir = project_root() / RAW_DIR
    out_dir = project_root() / CHUNKS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(raw_dir.glob("t_*.json"))
    if not raw_files:
        raise FileNotFoundError(f"No raw transcript files found in: {raw_dir}. Run 01_gather_all.py first.")

    all_chunks: List[Dict] = []

    for raw_file in raw_files:
        raw = json.loads(raw_file.read_text(encoding="utf-8"))
        video_id = raw.get("video_id", "unknown")
        segments = raw.get("segments", [])

        chunks = chunk_transcript(video_id, segments)
        all_chunks.extend(chunks)

        out_file = out_dir / raw_file.name.replace(".json", "_chunks.json")
        out_file.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"OK -> {out_file} | chunks: {len(chunks)}")

    merged_path = out_dir / "all_chunks.json"
    merged_path.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nMerged dataset -> {merged_path} | total chunks: {len(all_chunks)}")


if __name__ == "__main__":
    main()