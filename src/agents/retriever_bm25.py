"""
retriever_bm25.py

Keyword-based retriever using BM25.

Architecture intent:
- Build BM25 only from the canonical chunk dataset
- Use the same metadata schema as FAISS and Chroma
"""

# Enable postponed evaluation of type annotations.
from __future__ import annotations

# Standard library imports.
import json
from pathlib import Path
from typing import Any, Dict, List

# LangChain document abstraction used across the RAG pipeline.
from langchain_core.documents import Document

# BM25 retriever implementation from LangChain.
from langchain_community.retrievers import BM25Retriever


# Default path to the canonical chunk dataset used to build the BM25 retriever.
DEFAULT_CHUNKS_PATH = "data/chunks/all_chunks_stable.json"

# Default number of documents returned for each BM25 query.
DEFAULT_TOP_K = 5


def project_root() -> Path:
    """
    Resolve the root directory of the project.

    This assumes the file lives under:
        project_root/src/agents/retriever_bm25.py
    """
    return Path(__file__).resolve().parents[2]


def load_json(path: Path) -> Any:
    """
    Load and parse a JSON file from disk.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def build_documents(chunks: List[Dict[str, Any]]) -> List[Document]:
    """
    Convert chunk dictionaries into LangChain Document objects.

    Each chunk becomes a Document with:
    - page_content: the chunk text
    - metadata: normalized metadata aligned with the vector retrievers
    """
    docs: List[Document] = []

    for c in chunks:
        # Extract the chunk text and normalize surrounding whitespace.
        text = str(c.get("text", "")).strip()

        # Skip empty chunks because they are not useful for retrieval.
        if not text:
            continue

        # Keep the metadata schema consistent with FAISS and Chroma documents.
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

        # Create a LangChain Document for the current chunk.
        docs.append(
            Document(
                page_content=text,
                metadata=metadata,
            )
        )

    return docs


class RetrieverBM25:
    """
    Keyword-based retriever built from the canonical chunk dataset.

    This retriever is intended to complement semantic retrieval by capturing
    exact lexical matches, named entities, and keyword-heavy queries.
    """

    def __init__(
        self,
        chunks_path: str = DEFAULT_CHUNKS_PATH,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        """
        Initialize the BM25 retriever.

        Parameters
        ----------
        chunks_path : str
            Relative path to the canonical chunk dataset.

        top_k : int
            Maximum number of retrieved documents returned per query.
        """
        # Resolve the absolute path to the canonical chunk file.
        base = project_root()
        canonical_path = base / chunks_path

        # Ensure the source dataset exists before building the retriever.
        if not canonical_path.exists():
            raise FileNotFoundError(
                f"Canonical chunks file not found: {canonical_path}"
            )

        # Load the chunk dataset from disk.
        chunks = load_json(canonical_path)

        # Validate the structure of the loaded dataset.
        if not isinstance(chunks, list) or not chunks:
            raise ValueError("all_chunks_stable.json must be a non-empty list")

        # Convert chunk records into LangChain Document objects.
        docs = build_documents(chunks)

        # Ensure that at least one valid document was created.
        if not docs:
            raise ValueError("No valid documents were built for BM25")

        # Build the BM25 retriever from the canonical document collection.
        self.retriever = BM25Retriever.from_documents(
            docs,
            k=top_k,
            bm25_variant="plus",
        )

    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve documents relevant to the input query using BM25.
        """
        return self.retriever.invoke(query)