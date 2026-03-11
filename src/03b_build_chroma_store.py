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
from langchain_openai import OpenAIEmbeddings


# Path to the canonical chunk dataset used to build the Chroma store.
CHUNKS_PATH = "data/chunks/all_chunks_stable.json"

# Output directory where the persistent Chroma store will be saved.
CHROMA_DIR = "data/chroma"

# Name of the Chroma collection.
CHROMA_COLLECTION = "wwii_chunks"

# Default embedding model used when EMBED_MODEL is not defined in .env.
DEFAULT_EMBED_MODEL = "text-embedding-3-large"


def project_root() -> Path:
    """
    Resolve the project root directory.

    This assumes the file lives under:
        project_root/src/03b_build_chroma_store.py
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

    The metadata schema matches the one used in FAISS and retrieval so that
    all downstream components can operate consistently.
    """
    docs: List[Document] = []

    for c in chunks:
        # Read and normalize the chunk text.
        text = str(c.get("text", "")).strip()

        # Skip empty chunks because they are not useful for indexing.
        if not text:
            continue

        # Build the metadata dictionary for the LangChain document.
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

        # Convert the canonical chunk into a LangChain Document.
        docs.append(
            Document(
                page_content=text,
                metadata=metadata,
            )
        )

    return docs


def main() -> None:
    """
    Build the persistent Chroma vector store from the canonical chunk dataset.

    Pipeline:
    1. Load environment variables
    2. Validate API configuration
    3. Load the canonical chunk dataset
    4. Convert chunks into LangChain documents
    5. Create embeddings
    6. Build and persist the Chroma collection
    """
    # Resolve the project root and load environment variables.
    base = project_root()
    load_dotenv(base / ".env")

    # Ensure the OpenAI API key exists before creating embeddings.
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY in FINAL_PROJECT/.env")

    # Read the embedding model from .env, falling back to the default.
    embed_model = os.getenv("EMBED_MODEL", DEFAULT_EMBED_MODEL)

    # Resolve input and output paths.
    chunks_path = base / CHUNKS_PATH
    chroma_dir = base / CHROMA_DIR

    # Ensure the canonical chunk dataset exists.
    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing canonical chunks file: {chunks_path}")

    # Load and validate the canonical dataset.
    chunks: List[Dict[str, Any]] = load_json(chunks_path)
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("all_chunks_stable.json must be a non-empty list")

    # Convert canonical chunks into LangChain documents.
    docs = build_documents(chunks)

    # Ensure at least one valid document is available.
    if not docs:
        raise ValueError("No valid documents were built from all_chunks_stable.json")

    print(f"[INFO] Building Chroma from {len(docs)} docs")
    print(f"[INFO] Embedding model: {embed_model}")
    print(f"[INFO] Collection: {CHROMA_COLLECTION}")
    print(f"[INFO] Output directory: {chroma_dir}")

    # Initialize the embedding model used to vectorize document text.
    embeddings = OpenAIEmbeddings(model=embed_model)

    # Remove the previous Chroma store to avoid stale or duplicated content.
    if chroma_dir.exists():
        print(f"[INFO] Removing previous Chroma store: {chroma_dir}")
        shutil.rmtree(chroma_dir)

    # Import Chroma only when needed so the error message is clearer
    # if the dependency is missing.
    try:
        from langchain_chroma import Chroma
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "langchain_chroma is not installed. "
            "Install it with: python -m pip install langchain-chroma chromadb"
        ) from e

    # Create the persistent Chroma vector store.
    vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION,
        persist_directory=str(chroma_dir),
        embedding_function=embeddings,
    )

    # Index the dataset in batches to avoid overly large single insert operations.
    batch_size = 200

    for i in range(0, len(docs), batch_size):
        # Select the current batch of documents.
        batch = docs[i:i + batch_size]

        # Use chunk_id as the Chroma document ID whenever available.
        # This provides stable identifiers across rebuilds.
        ids = [
            doc.metadata.get("chunk_id") or f"chunk_{i + j}"
            for j, doc in enumerate(batch)
        ]

        # Insert the current batch into the Chroma collection.
        vectorstore.add_documents(batch, ids=ids)

        print(f"[INFO] Indexed {i + len(batch)} / {len(docs)}")

    print(f"[OK] Chroma store saved to: {chroma_dir}")
    print(f"[OK] Embedding model used: {embed_model}")
    print(f"[OK] Collection: {CHROMA_COLLECTION}")


# Run the script only when executed directly.
if __name__ == "__main__":
    main()