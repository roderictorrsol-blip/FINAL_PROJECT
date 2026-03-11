"""
context_builder.py

Context builder agent for RAG.

Responsibilities:
- Deduplicate retrieved documents
- Limit the number of final documents
- Build a structured context string for the LLM
- Extract citation URLs
"""

from __future__ import annotations

from typing import Dict, List

from langchain_core.documents import Document


DEFAULT_MAX_DOCS = 10


def normalize_text(text: str, limit: int = 400) -> str:
    """
    Normalize text for stronger duplicate detection.
    """
    text = (text or "").replace("\n", " ").strip()
    text = " ".join(text.split()).lower()
    return text[:limit]


class ContextBuilder:
    """
    Context builder for retrieved documents.

    Example:
        builder = ContextBuilder()
        result = builder.build(docs)
        context = result["context"]
        citations = result["citations"]
    """

    def __init__(self, max_docs: int = DEFAULT_MAX_DOCS) -> None:
        self.max_docs = max_docs

    @staticmethod
    def _doc_key(doc: Document) -> str:
        """
        Build a stable dedupe key from metadata or content.
        """
        m = doc.metadata or {}

        doc_id = m.get("doc_id")
        if doc_id:
            return f"doc_id:{doc_id}"

        chunk_id = m.get("chunk_id")
        if chunk_id:
            return f"chunk_id:{chunk_id}"

        source = m.get("source_url_t") or m.get("source_url") or "unknown_source"
        content_key = normalize_text(doc.page_content, limit=120)

        return f"{source}|{content_key}"

    def dedupe_docs(self, docs: List[Document]) -> List[Document]:
        """
        Deduplicate documents while preserving order.
        """
        seen = set()
        out: List[Document] = []

        for doc in docs:
            key = self._doc_key(doc)
            if key in seen:
                continue

            seen.add(key)
            out.append(doc)

            if len(out) >= self.max_docs:
                break

        return out

    @staticmethod
    def format_docs(docs: List[Document]) -> str:
        """
        Format documents into a structured context block for the LLM.
        """
        blocks = []

        for i, doc in enumerate(docs, start=1):
            m = doc.metadata or {}

            doc_id = m.get("doc_id", "unknown_doc")
            video_id = m.get("video_id", "unknown_video")
            video_title = m.get("video_title", "unknown_title")
            start_hhmmss = m.get("start_hhmmss", "unknown")
            end_hhmmss = m.get("end_hhmmss", "unknown")
            source_url = m.get("source_url_t") or m.get("source_url") or "unknown_source"

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

        return "\n".join(blocks)

    @staticmethod
    def extract_citations(docs: List[Document]) -> List[str]:
        """
        Extract unique citation URLs from the documents.
        """
        seen = set()
        out: List[str] = []

        for doc in docs:
            m = doc.metadata or {}
            url = m.get("source_url_t") or m.get("source_url")
            if url and url not in seen:
                seen.add(url)
                out.append(url)

        return out

    def build(self, docs: List[Document]) -> Dict[str, object]:
        """
        Build final RAG context package from raw retrieved docs.
        """
        final_docs = self.dedupe_docs(docs)
        context = self.format_docs(final_docs)
        citations = self.extract_citations(final_docs)

        return {
            "docs": final_docs,
            "context": context,
            "citations": citations,
        }