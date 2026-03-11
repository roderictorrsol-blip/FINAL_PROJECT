"""
03b_build_chroma_store.py

Build a persistent Chroma store from the canonical chunk dataset:
- data/chunks/all_chunks_stable.json

Output:
- data/chroma/

Architecture intent:
- Chroma = persistent source of truth
- FAISS = fast snapshot index built from the same canonical data
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


CHUNKS_PATH = "data/chunks/all_chunks_stable.json"
CHROMA_DIR = "data/chroma"
CHROMA_COLLECTION = "wwii_chunks"
DEFAULT_EMBED_MODEL = "text-embedding-3-large"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_documents(chunks: List[Dict[str, Any]]) -> List[Document]:
    docs: List[Document] = []

    for c in chunks:
        text = str(c.get("text", "")).strip()
        if not text:
            continue

        metadata = {
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
        }

        docs.append(
            Document(
                page_content=text,
                metadata=metadata,
            )
        )

    return docs


def main() -> None:
    base = project_root()
    load_dotenv(base / ".env")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY in FINAL_PROJECT/.env")

    embed_model = os.getenv("EMBED_MODEL", DEFAULT_EMBED_MODEL)

    chunks_path = base / CHUNKS_PATH
    chroma_dir = base / CHROMA_DIR

    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing canonical chunks file: {chunks_path}")

    chunks: List[Dict[str, Any]] = load_json(chunks_path)
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("all_chunks_stable.json must be a non-empty list")

    docs = build_documents(chunks)
    if not docs:
        raise ValueError("No valid documents were built from all_chunks_stable.json")

    print(f"[INFO] Building Chroma from {len(docs)} docs")
    print(f"[INFO] Embedding model: {embed_model}")
    print(f"[INFO] Collection: {CHROMA_COLLECTION}")
    print(f"[INFO] Output directory: {chroma_dir}")

    embeddings = OpenAIEmbeddings(model=embed_model)

    # Remove previous Chroma store to avoid stale duplicated content
    if chroma_dir.exists():
        print(f"[INFO] Removing previous Chroma store: {chroma_dir}")
        shutil.rmtree(chroma_dir)

    try:
        from langchain_chroma import Chroma
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "langchain_chroma is not installed. "
            "Install it with: python -m pip install langchain-chroma chromadb"
        ) from e

    vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION,
        persist_directory=str(chroma_dir),
        embedding_function=embeddings,
    )

    batch_size = 200

    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        ids = [
            doc.metadata.get("chunk_id") or f"chunk_{i + j}"
            for j, doc in enumerate(batch)
        ]

        vectorstore.add_documents(batch, ids=ids)
        print(f"[INFO] Indexed {i + len(batch)} / {len(docs)}")

    print(f"[OK] Chroma store saved to: {chroma_dir}")
    print(f"[OK] Embedding model used: {embed_model}")
    print(f"[OK] Collection: {CHROMA_COLLECTION}")


if __name__ == "__main__":
    main()