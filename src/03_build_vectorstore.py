"""
03_build_vectorstore.py

Build a LangChain FAISS vectorstore using OpenAIEmbeddings from:
- data/chunks/all_chunks_stable.json

Output:
- data/vectorstore/faiss_store_openai/
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


CHUNKS_PATH = "data/chunks/all_chunks_stable.json"
OUT_STORE_DIR = "data/vectorstore/faiss_store_openai"
EMBED_MODEL = "text-embedding-3-small"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    base = project_root()
    load_dotenv(base / ".env")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY in FINAL_PROJECT/.env")

    chunks_path = base / CHUNKS_PATH
    store_dir = base / OUT_STORE_DIR

    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing {chunks_path}")

    chunks: List[Dict[str, Any]] = load_json(chunks_path)
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("all_chunks_stable.json must be a non-empty list")

    docs: List[Document] = []

    for c in chunks:
        text = str(c.get("text", "")).strip()
        if not text:
            continue

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "doc_id": c.get("doc_id"),
                    "video_id": c.get("video_id"),
                    "video_title": c.get("video_title"),
                    "thumbnail_url": c.get("thumbnail_url"),
                    "chunk_id": c.get("chunk_id"),
                    "start": c.get("start"),
                    "end": c.get("end"),
                    "start_hhmmss": c.get("start_hhmmss"),
                    "end_hhmmss": c.get("end_hhmmss"),
                    "source_url": c.get("source_url"),
                    "source_url_t": c.get("source_url_t"),
                },
            )
        )

    print(f"Building FAISS from {len(docs)} docs...")
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)

    vs = FAISS.from_documents(docs, embeddings)

    store_dir.parent.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(store_dir))

    print(f"OK -> {store_dir}")
    print(f"Embedding model: {EMBED_MODEL}")


if __name__ == "__main__":
    main()