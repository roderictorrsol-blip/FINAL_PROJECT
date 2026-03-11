"""
02_build_canonical_chunks.py

Build the canonical chunk dataset for the WWII RAG project.

This script replaces the old multi-step pipeline that previously required:
- transcript chunking
- stable doc_id generation
- URL normalization
- metadata enrichment (thumbnail + video title)

Inputs:
- data/raw/t_*.json

Outputs:
- data/chunks/t_01_chunks.json
- data/chunks/t_02_chunks.json
- ...
- data/chunks/all_chunks_stable.json

Each final chunk includes:
- chunk_id
- doc_id
- video_id
- video_title
- thumbnail_url
- text
- text_norm
- start / end
- start_hhmmss / end_hhmmss
- source_url
- source_url_t
"""

# Enable postponed evaluation of type annotations.
from __future__ import annotations

# Standard library imports.
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import parse_qsl, quote_plus, urlencode, urlparse, urlunparse
from urllib.request import urlopen


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

# Directory containing raw transcript JSON files.
RAW_DIR = "data/raw"

# Directory where per-video chunk files are written.
CHUNKS_DIR = "data/chunks"

# Path of the final merged canonical dataset.
OUT_STABLE_JSON = "data/chunks/all_chunks_stable.json"


# ---------------------------------------------------------
# Chunking configuration
# ---------------------------------------------------------

# Maximum approximate size of each text chunk in characters.
MAX_CHARS = 300

# Number of overlapping characters between consecutive chunks.
OVERLAP_CHARS = 100


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def project_root() -> Path:
    """
    Return the project root directory.

    This assumes the file lives under:
        project_root/src/02_build_canonical_chunks.py
    """
    return Path(__file__).resolve().parents[1]


def load_json(path: Path) -> Any:
    """
    Load and parse a JSON file from disk.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    """
    Save a JSON payload to disk using UTF-8 and pretty formatting.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def format_hhmmss(seconds: float) -> str:
    """
    Format seconds into mm:ss or hh:mm:ss.
    """
    sec = int(seconds)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def normalize_text(text: str, limit: int = 500) -> str:
    """
    Normalize text for stable hashing and cleaner metadata.

    This helper:
    - replaces line breaks with spaces
    - strips outer whitespace
    - collapses repeated whitespace
    - truncates the result to a fixed limit
    """
    text = (text or "").replace("\n", " ").strip()
    text = " ".join(text.split())
    return text[:limit]


def stable_doc_id(video_id: str, start: float, end: float, text: str) -> str:
    """
    Build a stable document identifier from key chunk properties.

    The hash is based on:
    - video_id
    - start timestamp
    - end timestamp
    - normalized text prefix

    This allows deterministic doc_id generation across repeated runs.
    """
    key = f"{video_id}|{start:.2f}|{end:.2f}|{normalize_text(text, 160)}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return f"{video_id}_{int(start):06d}_{digest}"


def canonical_youtube_url(video_id: str) -> str:
    """
    Build the canonical YouTube URL for a video.
    """
    return f"https://www.youtube.com/watch?v={video_id}"


def strip_time_params(url: str) -> str:
    """
    Remove time-related query parameters from a YouTube URL.

    This keeps the base video URL clean before adding a canonical timestamp.
    """
    parsed = urlparse(url)
    query_params = dict(parse_qsl(parsed.query, keep_blank_values=True))

    query_params.pop("t", None)
    query_params.pop("start", None)
    query_params.pop("time_continue", None)

    cleaned_query = urlencode(query_params, doseq=True)

    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            cleaned_query,
            parsed.fragment,
        )
    )


def add_timestamp_url(base_url: str, start_seconds: float) -> str:
    """
    Add a canonical timestamp parameter to a YouTube URL.

    Example:
        https://www.youtube.com/watch?v=abc123
        ->
        https://www.youtube.com/watch?v=abc123&t=42s
    """
    clean_url = strip_time_params(base_url)
    sep = "&" if "?" in clean_url else "?"
    return f"{clean_url}{sep}t={int(start_seconds)}s"


def fetch_youtube_title(source_url: str) -> str | None:
    """
    Resolve the real YouTube title through the oEmbed endpoint.

    Returns None if the request fails.
    """
    try:
        oembed_url = (
            "https://www.youtube.com/oembed"
            f"?url={quote_plus(source_url)}&format=json"
        )

        with urlopen(oembed_url, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))

        return payload.get("title")
    except Exception:
        return None


# ---------------------------------------------------------
# Transcript processing
# ---------------------------------------------------------

def build_text_and_mapping(
    segments: List[Dict[str, Any]],
) -> Tuple[str, List[Tuple[int, int, float, float]]]:
    """
    Build a concatenated transcript string and a mapping from text spans
    to original segment timestamps.

    Returns:
    - full_text
    - mapping: (start_idx, end_idx, seg_start, seg_end)
    """
    # Accumulate transcript text pieces.
    pieces: List[str] = []

    # Store character span to time mappings for each original segment.
    mapping: List[Tuple[int, int, float, float]] = []

    # Running character index in the concatenated transcript.
    idx = 0

    for seg in segments:
        # Read the segment text.
        text = (seg.get("text") or "").strip()

        # Skip empty transcript segments.
        if not text:
            continue

        # Read timing information from the raw transcript segment.
        start = float(seg.get("start", 0.0))
        duration = float(seg.get("duration", 0.0))
        end = start + duration

        # Add a trailing space to reduce the risk of word concatenation.
        chunk_text = text.replace("\n", " ").strip() + " "

        # Compute the character span covered by this transcript segment.
        i0 = idx
        i1 = idx + len(chunk_text)

        # Append text and store the character-to-time mapping.
        pieces.append(chunk_text)
        mapping.append((i0, i1, start, end))
        idx = i1

    # Return the full transcript string plus the mapping metadata.
    return "".join(pieces).strip(), mapping


def span_to_time(
    a: int,
    b: int,
    mapping: List[Tuple[int, int, float, float]],
) -> Tuple[float, float]:
    """
    Convert a text span [a, b) to timestamps by looking at overlapping segments.
    """
    start_t = None
    end_t = None

    for i0, i1, t0, t1 in mapping:
        # The first overlapping segment determines the start time.
        if start_t is None and i1 > a:
            start_t = t0

        # The last overlapping segment determines the end time.
        if i0 < b:
            end_t = t1

    # Fallbacks for edge cases.
    if start_t is None:
        start_t = 0.0
    if end_t is None:
        end_t = start_t

    return float(start_t), float(end_t)


def chunk_transcript(video_id: str, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Chunk a transcript into RAG-friendly passages with overlap.

    Each chunk includes:
    - chunk_id
    - video_id
    - text
    - start/end timestamps
    - human-readable timestamps
    """
    # Build the full transcript and the character-to-time mapping.
    full_text, mapping = build_text_and_mapping(segments)
    n = len(full_text)

    # Final list of produced chunks.
    chunks: List[Dict[str, Any]] = []

    # Current cursor in the concatenated transcript text.
    i = 0

    while i < n:
        # Propose a chunk end based on maximum chunk length.
        j = min(i + MAX_CHARS, n)

        # Prefer splitting at whitespace to avoid cutting words in the middle.
        if j < n:
            cut = full_text.rfind(" ", i, j)
            if cut != -1 and cut > i + 200:
                j = cut

        # Convert the selected text span into timestamps.
        start_t, end_t = span_to_time(i, j, mapping)

        # Extract the current chunk text.
        chunk_text = full_text[i:j].strip()

        # Keep only non-empty chunks.
        if chunk_text:
            chunks.append(
                {
                    "chunk_id": f"{video_id}_{len(chunks):04d}",
                    "video_id": video_id,
                    "text": chunk_text,
                    "start": start_t,
                    "end": end_t,
                    "start_hhmmss": format_hhmmss(start_t),
                    "end_hhmmss": format_hhmmss(end_t),
                }
            )

        # Advance with overlap to preserve context between adjacent chunks.
        next_i = j - OVERLAP_CHARS
        i = next_i if next_i > i else j
        if i <= 0:
            i = j

    return chunks


# ---------------------------------------------------------
# Canonical metadata enrichment
# ---------------------------------------------------------

def enrich_chunk(
    chunk: Dict[str, Any],
    titles_cache: Dict[str, str],
) -> Dict[str, Any]:
    """
    Add canonical metadata to a chunk.

    This includes:
    - stable doc_id
    - canonical base URL
    - timestamped URL
    - normalized text
    - thumbnail URL
    - resolved video title
    """
    # Read the main chunk properties.
    video_id = str(chunk.get("video_id", "unknown"))
    text = str(chunk.get("text", ""))
    start = float(chunk.get("start", 0.0))
    end = float(chunk.get("end", start))

    # Build canonical source URLs.
    source_url = canonical_youtube_url(video_id)
    source_url_t = add_timestamp_url(source_url, start)

    # Normalize text for downstream indexing/debugging.
    text_norm = normalize_text(text, 500)

    # Build the stable document identifier.
    doc_id = stable_doc_id(video_id, start, end, text)

    # Build a direct thumbnail URL from the YouTube video ID.
    thumbnail_url = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"

    # Resolve and cache the video title to avoid repeated oEmbed calls.
    if video_id in titles_cache:
        video_title = titles_cache[video_id]
    else:
        fetched_title = fetch_youtube_title(source_url)
        video_title = fetched_title or video_id
        titles_cache[video_id] = video_title

    # Copy the original chunk and enrich it with canonical metadata.
    enriched = dict(chunk)
    enriched["doc_id"] = doc_id
    enriched["source_url"] = source_url
    enriched["source_url_t"] = source_url_t
    enriched["text_norm"] = text_norm
    enriched["thumbnail_url"] = thumbnail_url
    enriched["video_title"] = video_title

    return enriched


def deduplicate_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate chunks using doc_id while preserving order.
    """
    seen = set()
    unique_chunks: List[Dict[str, Any]] = []

    for chunk in chunks:
        doc_id = chunk.get("doc_id")

        # Skip repeated canonical documents.
        if doc_id in seen:
            continue

        seen.add(doc_id)
        unique_chunks.append(chunk)

    return unique_chunks


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:
    """
    Build the full canonical chunk dataset from raw transcript files.

    Pipeline:
    1. Load raw transcript JSON files
    2. Chunk each transcript
    3. Enrich chunks with stable IDs and metadata
    4. Save per-video chunk files
    5. Merge and deduplicate all chunks
    6. Save the final canonical dataset
    """
    # Resolve the main input/output paths.
    base = project_root()
    raw_dir = base / RAW_DIR
    chunks_dir = base / CHUNKS_DIR
    out_stable_path = base / OUT_STABLE_JSON

    # Ensure the output directory exists.
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # Discover all raw transcript files.
    raw_files = sorted(raw_dir.glob("t_*.json"))
    if not raw_files:
        raise FileNotFoundError(
            f"No raw transcript files found in: {raw_dir}. "
            "Run the transcript gathering step first."
        )

    # Store all enriched chunks across all videos.
    all_chunks: List[Dict[str, Any]] = []

    # Cache titles so each video title is fetched at most once.
    titles_cache: Dict[str, str] = {}

    # Process each raw transcript file independently.
    for raw_file in raw_files:
        raw = load_json(raw_file)

        # Read the source video id and transcript segments.
        video_id = raw.get("video_id", "unknown")
        segments = raw.get("segments", [])

        # Chunk the transcript and enrich each chunk with canonical metadata.
        chunks = chunk_transcript(video_id, segments)
        enriched_chunks = [enrich_chunk(chunk, titles_cache) for chunk in chunks]

        # Save the per-video chunk file.
        out_file = chunks_dir / raw_file.name.replace(".json", "_chunks.json")
        save_json(out_file, enriched_chunks)

        # Add chunks to the merged canonical collection.
        all_chunks.extend(enriched_chunks)

        print(f"[OK] {out_file} | chunks: {len(enriched_chunks)}")

    # Deduplicate the merged chunk list using stable doc_id.
    all_chunks = deduplicate_chunks(all_chunks)

    # Save the final canonical dataset.
    save_json(out_stable_path, all_chunks)

    print()
    print(f"[OK] Canonical dataset saved to: {out_stable_path}")
    print(f"[OK] Total canonical chunks: {len(all_chunks)}")
    print(f"[OK] Unique videos processed: {len(titles_cache)}")


# Run the script only when executed directly.
if __name__ == "__main__":
    main()