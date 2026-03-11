"""
03_build_vectorstore.py

Build a LangChain FAISS vectorstore using OpenAIEmbeddings from:
- data/chunks/all_chunks_stable.json

Output:
- data/vectorstore/faiss_store_openai/

Notes:
- The embedding model used here must match the one used at retrieval time.
- Recommended architecture:
    - Chroma = persistent source of truth
    - FAISS = fast snapshot index
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


CHUNKS_PATH = "data/chunks/all_chunks_stable.json"
OUT_STORE_DIR = "data/vectorstore/faiss_store_openai"
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

    return docs


def main() -> None:
    base = project_root()
    load_dotenv(base / ".env")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY in FINAL_PROJECT/.env")

    embed_model = os.getenv("EMBED_MODEL", DEFAULT_EMBED_MODEL)

    chunks_path = base / CHUNKS_PATH
    store_dir = base / OUT_STORE_DIR

    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing {chunks_path}")

    chunks: List[Dict[str, Any]] = load_json(chunks_path)
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("all_chunks_stable.json must be a non-empty list")

    docs = build_documents(chunks)

    if not docs:
        raise ValueError("No valid documents were built from all_chunks_stable.json")

    print(f"[INFO] Building FAISS from {len(docs)} docs")
    print(f"[INFO] Embedding model: {embed_model}")
    print(f"[INFO] Output directory: {store_dir}")

    embeddings = OpenAIEmbeddings(model=embed_model)

    # Remove previous FAISS store to avoid mixing incompatible index dimensions
    if store_dir.exists():
        print(f"[INFO] Removing previous FAISS store: {store_dir}")
        shutil.rmtree(store_dir)

    vs = FAISS.from_documents(docs, embeddings)

    store_dir.parent.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(store_dir))

    print(f"[OK] FAISS store saved to: {store_dir}")
    print(f"[OK] Embedding model used: {embed_model}")


if __name__ == "__main__":
    main()