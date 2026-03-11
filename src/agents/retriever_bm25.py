"""
retriever_bm25.py

Keyword-based retriever using BM25.

Architecture intent:
- Build BM25 only from the canonical chunk dataset
- Use the same metadata schema as FAISS and Chroma
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever


DEFAULT_CHUNKS_PATH = "data/chunks/all_chunks_stable.json"
DEFAULT_TOP_K = 5


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


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


class RetrieverBM25:
    def __init__(
        self,
        chunks_path: str = DEFAULT_CHUNKS_PATH,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        base = project_root()
        canonical_path = base / chunks_path

        if not canonical_path.exists():
            raise FileNotFoundError(
                f"Canonical chunks file not found: {canonical_path}"
            )

        chunks = load_json(canonical_path)

        if not isinstance(chunks, list) or not chunks:
            raise ValueError("all_chunks_stable.json must be a non-empty list")

        docs = build_documents(chunks)

        if not docs:
            raise ValueError("No valid documents were built for BM25")

        self.retriever = BM25Retriever.from_documents(
            docs,
            k=top_k,
            bm25_variant="plus",
        )

    def retrieve(self, query: str) -> List[Document]:
        return self.retriever.invoke(query)