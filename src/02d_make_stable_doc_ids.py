"""
02d_make_stable_doc_ids.py

Add stable doc_id and clean timestamped source_url_t to chunks.

Input:
- data/chunks/all_chunks.json

Output:
- data/chunks/all_chunks_stable.json

doc_id strategy:
- stable hash from (video_id, start, end, normalized_text_prefix)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse


IN_PATH = "data/chunks/all_chunks.json"
OUT_JSON = "data/chunks/all_chunks_stable.json"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_text(s: str, limit: int = 200) -> str:
    s = (s or "").replace("\n", " ").strip()
    s = " ".join(s.split())
    return s[:limit]


def stable_id(video_id: str, start: float, end: float, text: str) -> str:
    """
    Build a stable id from key properties.
    We round timestamps to 2 decimals to avoid float noise.
    """
    key = f"{video_id}|{start:.2f}|{end:.2f}|{normalize_text(text, 160)}"
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return f"{video_id}_{int(start):06d}_{h}"


def canonical_youtube_url(video_id: str) -> str:
    """
    Return the canonical base YouTube URL for a video.
    """
    return f"https://www.youtube.com/watch?v={video_id}"


def strip_time_params(url: str) -> str:
    """
    Remove time-related query parameters from a YouTube URL.
    Keeps other query params only if needed, but in this project
    we mainly want a clean canonical video URL.
    """
    parsed = urlparse(url)
    query_params = dict(parse_qsl(parsed.query, keep_blank_values=True))

    # Remove time-related params
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
    Create a YouTube URL with a single start time parameter.
    """
    clean_url = strip_time_params(base_url)
    sep = "&" if "?" in clean_url else "?"
    return f"{clean_url}{sep}t={int(start_seconds)}s"


def main() -> None:
    base = project_root()
    in_path = base / IN_PATH

    if not in_path.exists():
        raise FileNotFoundError(f"Missing {in_path}")

    chunks: List[Dict[str, Any]] = load_json(in_path)
    if not isinstance(chunks, list):
        raise ValueError("all_chunks.json must be a list[dict]")

    out_chunks: List[Dict[str, Any]] = []

    for c in chunks:
        video_id = str(c.get("video_id", "unknown"))
        text = str(c.get("text", ""))
        start = float(c.get("start", 0.0))
        end = float(c.get("end", start))

        doc_id = stable_id(video_id, start, end, text)

        # Always normalize to canonical base URL
        source_url = canonical_youtube_url(video_id)
        source_url_t = add_timestamp_url(source_url, start)

        c2 = dict(c)
        c2["doc_id"] = doc_id
        c2["source_url"] = source_url
        c2["source_url_t"] = source_url_t
        c2["text_norm"] = normalize_text(text, 500)
        out_chunks.append(c2)

    out_json_path = base / OUT_JSON
    save_json(out_json_path, out_chunks)

    print(f"OK -> {out_json_path} | chunks: {len(out_chunks)}")


if __name__ == "__main__":
    main()