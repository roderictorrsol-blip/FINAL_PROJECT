from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote_plus
from urllib.request import urlopen


CHUNKS_PATH = Path("data/chunks/all_chunks_stable.json")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data):
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def fetch_youtube_title(source_url: str) -> str | None:
    """
    Try obtain real video title using oEmbed.
    If fails, return None.
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


def main():
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Unexisting: {CHUNKS_PATH}")

    chunks = load_json(CHUNKS_PATH)

    if not isinstance(chunks, list):
        raise ValueError("Chunk´s file must contain a list")

    titles_cache: dict[str, str] = {}
    updated = 0

    for chunk in chunks:
        video_id = chunk.get("video_id")
        source_url = chunk.get("source_url")

        if not video_id:
            continue

        # Direct miniature from video_id
        chunk["thumbnail_url"] = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"

        # Reuse title already resolved for that video
        if video_id in titles_cache:
            chunk["video_title"] = titles_cache[video_id]
            updated += 1
            continue

        title = None
        if source_url:
            title = fetch_youtube_title(source_url)

        # Fallback: using video_id if title not found
        final_title = title or video_id

        chunk["video_title"] = final_title
        titles_cache[video_id] = final_title
        updated += 1

    save_json(CHUNKS_PATH, chunks)

    print(f"OK -> added/updated metadata in {CHUNKS_PATH}")
    print(f"updated chunks: {updated}")
    print(f"unique videos porcessed: {len(titles_cache)}")


if __name__ == "__main__":
    main()