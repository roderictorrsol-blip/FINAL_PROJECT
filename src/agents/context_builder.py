"""
context_builder.py

Context builder agent for RAG.

Responsibilities:
- Deduplicate retrieved documents
- Limit the number of final documents
- Build a structured context string for the LLM
- Extract citation URLs
"""

# Enable postponed evaluation of type annotations.
from __future__ import annotations

# Typing utilities used for structured return values and lists.
from typing import Dict, List

# LangChain document abstraction used across the pipeline.
from langchain_core.documents import Document


# Default maximum number of documents included in the final context package.
DEFAULT_MAX_DOCS = 10


def normalize_text(text: str, limit: int = 400) -> str:
    """
    Normalize text for stronger duplicate detection.

    This helper reduces formatting noise by:
    - replacing line breaks with spaces
    - collapsing repeated whitespace
    - lowercasing the text
    - truncating the result to a fixed limit
    """
    text = (text or "").replace("\n", " ").strip()
    text = " ".join(text.split()).lower()
    return text[:limit]


class ContextBuilder:
    """
    Context builder for retrieved documents.

    This agent prepares the final context package used by the answer generator.
    It:
    - removes duplicate documents
    - limits the number of documents
    - formats the context into a structured text block
    - extracts citation URLs

    Example:
        builder = ContextBuilder()
        result = builder.build(docs)
        context = result["context"]
        citations = result["citations"]
    """

    def __init__(self, max_docs: int = DEFAULT_MAX_DOCS) -> None:
        """
        Initialize the context builder.

        Parameters
        ----------
        max_docs : int
            Maximum number of unique documents kept in the final context.
        """
        self.max_docs = max_docs

    @staticmethod
    def _doc_key(doc: Document) -> str:
        """
        Build a stable deduplication key from metadata or content.

        Priority:
        1. doc_id
        2. chunk_id
        3. source URL + normalized content snippet

        This strategy ensures that documents can still be deduplicated even if
        some metadata fields are missing.
        """
        # Read document metadata safely.
        m = doc.metadata or {}

        # Prefer the most stable document-level identifier.
        doc_id = m.get("doc_id")
        if doc_id:
            return f"doc_id:{doc_id}"

        # Fall back to the chunk identifier when available.
        chunk_id = m.get("chunk_id")
        if chunk_id:
            return f"chunk_id:{chunk_id}"

        # As a final fallback, combine source URL and normalized content.
        source = m.get("source_url_t") or m.get("source_url") or "unknown_source"
        content_key = normalize_text(doc.page_content, limit=120)

        return f"{source}|{content_key}"

    def dedupe_docs(self, docs: List[Document]) -> List[Document]:
        """
        Deduplicate documents while preserving order.

        The first occurrence of each document is kept.
        The process stops once max_docs unique documents have been collected.
        """
        seen = set()
        out: List[Document] = []

        for doc in docs:
            # Build a stable key for duplicate detection.
            key = self._doc_key(doc)

            # Skip documents already seen.
            if key in seen:
                continue

            # Keep the first unique occurrence.
            seen.add(key)
            out.append(doc)

            # Stop when the configured maximum is reached.
            if len(out) >= self.max_docs:
                break

        return out

    @staticmethod
    def format_docs(docs: List[Document]) -> str:
        """
        Format documents into a structured context block for the LLM.

        Each document becomes a labeled source block containing:
        - document identifiers
        - video metadata
        - time range
        - source URL
        - raw text content
        """
        blocks = []

        for i, doc in enumerate(docs, start=1):
            # Read document metadata safely.
            m = doc.metadata or {}

            # Extract metadata fields with fallbacks for robustness.
            doc_id = m.get("doc_id", "unknown_doc")
            video_id = m.get("video_id", "unknown_video")
            video_title = m.get("video_title", "unknown_title")
            start_hhmmss = m.get("start_hhmmss", "unknown")
            end_hhmmss = m.get("end_hhmmss", "unknown")
            source_url = m.get("source_url_t") or m.get("source_url") or "unknown_source"

            # Build a structured source block for the LLM context window.
            block = (
                f"[SOURCE {i}]\n"
                f"doc_id: {doc_id}\n"
                f"video_id: {video_id}\n"
                f"video_title: {video_title}\n"
                f"time_range: {start_hhmmss} - {end_hhmmss}\n"
                f"source_url: {source_url}\n"
                f"text: {doc.page_content}\n"
            )

            blocks.append(block)

        # Join all source blocks into a single context string.
        return "\n".join(blocks)

    @staticmethod
    def extract_citations(docs: List[Document]) -> List[str]:
        """
        Extract unique citation URLs from the documents.

        URL priority:
        1. source_url_t
        2. source_url
        """
        seen = set()
        out: List[str] = []

        for doc in docs:
            # Read metadata safely.
            m = doc.metadata or {}

            # Prefer timestamped URLs when available.
            url = m.get("source_url_t") or m.get("source_url")

            # Keep only unique non-empty URLs.
            if url and url not in seen:
                seen.add(url)
                out.append(url)

        return out

    def build(self, docs: List[Document]) -> Dict[str, object]:
        """
        Build the final RAG context package from retrieved documents.

        Returns
        -------
        Dict[str, object]
            Dictionary containing:
            - docs: deduplicated final documents
            - context: formatted context string for the LLM
            - citations: unique citation URLs
        """
        # Deduplicate and limit the retrieved documents.
        final_docs = self.dedupe_docs(docs)

        # Convert the selected documents into the context string.
        context = self.format_docs(final_docs)

        # Extract unique citation URLs for UI display and prompt support.
        citations = self.extract_citations(final_docs)

        # Return the full context package.
        return {
            "docs": final_docs,
            "context": context,
            "citations": citations,
        }