from __future__ import annotations

import json
from pathlib import Path


CHUNKS_PATH = Path("data/chunks/all_chunks_stable.json")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    chunks = load_json(CHUNKS_PATH)

    for chunk in chunks:
        video_id = chunk.get("video_id")

        if video_id:
            # youtube´s minaiture
            chunk["thumbnail_url"] = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"

        # using id as title fallback
        chunk["video_title"] = video_id

    save_json(CHUNKS_PATH, chunks)

    print("✔ Metadata added to chunks")
    print(f"updated file: {CHUNKS_PATH}")


if __name__ == "__main__":
    main()