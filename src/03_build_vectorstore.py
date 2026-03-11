"""
03_build_vectorstore.py

Build a LangChain FAISS vectorstore using OpenAI embeddings from:
- data/chunks/all_chunks_stable.json

Output:
- data/vectorstore/faiss_store_openai/

Notes:
- The embedding model used here must match the one used at retrieval time.
- Recommended architecture:
    - Chroma = persistent source of truth
    - FAISS = fast snapshot index
"""

# Enable postponed evaluation of type annotations.
from __future__ import annotations

# Standard library imports.
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List

# Third-party imports.
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


# Path to the canonical chunk dataset used to build the FAISS index.
CHUNKS_PATH = "data/chunks/all_chunks_stable.json"

# Output directory where the FAISS index will be saved.
OUT_STORE_DIR = "data/vectorstore/faiss_store_openai"

# Default embedding model used when EMBED_MODEL is not defined in .env.
DEFAULT_EMBED_MODEL = "text-embedding-3-large"


def project_root() -> Path:
    """
    Resolve the project root directory.

    This assumes the file lives under:
        project_root/src/03_build_vectorstore.py
    """
    return Path(__file__).resolve().parents[1]


def load_json(path: Path) -> Any:
    """
    Load and parse a JSON file from disk.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def build_documents(chunks: List[Dict[str, Any]]) -> List[Document]:
    """
    Convert canonical chunk dictionaries into LangChain Document objects.

    Each document includes:
    - page_content: the chunk text
    - metadata: identifiers, timestamps, video metadata, and source URLs

    The metadata schema is aligned with the retrieval and citation layers
    so that downstream agents can operate consistently.
    """
    docs: List[Document] = []

    for c in chunks:
        # Read and normalize the main text content.
        text = str(c.get("text", "")).strip()

        # Skip empty chunks because they are not useful for embeddings.
        if not text:
            continue

        # Convert the canonical chunk into a LangChain Document.
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
    """
    Build the FAISS vector store from the canonical chunk dataset.

    Pipeline:
    1. Load environment variables
    2. Validate API configuration
    3. Load the canonical chunk dataset
    4. Convert chunks into LangChain documents
    5. Create embeddings
    6. Build and save the FAISS index
    """
    # Resolve the project root and load environment variables.
    base = project_root()
    load_dotenv(base / ".env")

    # Ensure the OpenAI API key is available before building embeddings.
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY in FINAL_PROJECT/.env")

    # Read the embedding model from .env, falling back to the default.
    embed_model = os.getenv("EMBED_MODEL", DEFAULT_EMBED_MODEL)

    # Resolve input and output paths.
    chunks_path = base / CHUNKS_PATH
    store_dir = base / OUT_STORE_DIR

    # Ensure the canonical chunk dataset exists.
    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing {chunks_path}")

    # Load and validate the canonical dataset.
    chunks: List[Dict[str, Any]] = load_json(chunks_path)
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("all_chunks_stable.json must be a non-empty list")

    # Convert canonical chunks into LangChain documents.
    docs = build_documents(chunks)

    # Ensure at least one valid document is available.
    if not docs:
        raise ValueError("No valid documents were built from all_chunks_stable.json")

    print(f"[INFO] Building FAISS from {len(docs)} docs")
    print(f"[INFO] Embedding model: {embed_model}")
    print(f"[INFO] Output directory: {store_dir}")

    # Initialize the embedding model used to vectorize document text.
    embeddings = OpenAIEmbeddings(model=embed_model)

    # Remove the previous FAISS store to avoid mixing indexes built
    # with incompatible embedding models or vector dimensions.
    if store_dir.exists():
        print(f"[INFO] Removing previous FAISS store: {store_dir}")
        shutil.rmtree(store_dir)

    # Build the FAISS vector store from the documents.
    vs = FAISS.from_documents(docs, embeddings)

    # Ensure the output parent directory exists.
    store_dir.parent.mkdir(parents=True, exist_ok=True)

    # Save the FAISS index locally.
    vs.save_local(str(store_dir))

    print(f"[OK] FAISS store saved to: {store_dir}")
    print(f"[OK] Embedding model used: {embed_model}")


# Run the script only when executed directly.
if __name__ == "__main__":
    main()