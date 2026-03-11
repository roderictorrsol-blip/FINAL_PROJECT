"""
reranker_agent.py

Reranker agent for RAG.

Responsibilities:
- Score retrieved documents against the user query
- Reorder documents by semantic relevance
- Apply light deduplication to improve result diversity
- Return the top-ranked documents
"""

# Enable postponed evaluation of type annotations.
from __future__ import annotations

# Typing utilities for document collections.
from typing import List

# LangChain document abstraction used across the pipeline.
from langchain_core.documents import Document

# Sentence Transformers cross-encoder used for reranking.
from sentence_transformers import CrossEncoder


# Default cross-encoder model used for document reranking.
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def normalize(text: str, limit: int = 400) -> str:
    """
    Normalize text for lightweight duplicate detection.

    This helper:
    - replaces line breaks with spaces
    - collapses repeated whitespace
    - lowercases the text
    - truncates the result to a fixed length
    """
    text = (text or "").replace("\n", " ").strip()
    text = " ".join(text.split()).lower()
    return text[:limit]


class RerankerAgent:
    """
    Rerank retrieved documents using a cross-encoder model.

    This agent improves retrieval quality by:
    - scoring each document against the original query
    - sorting documents by predicted relevance
    - removing near-duplicate content
    - keeping only the top-N most relevant documents
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANK_MODEL,
        top_n: int = 5,
    ) -> None:
        """
        Initialize the reranker.

        Parameters
        ----------
        model_name : str
            Name of the cross-encoder model used for reranking.

        top_n : int
            Maximum number of final documents returned.
        """
        # Load the cross-encoder model used to score query-document pairs.
        self.model = CrossEncoder(model_name)

        # Store the maximum number of reranked documents to keep.
        self.top_n = top_n

    def rerank(self, query: str, docs: List[Document]) -> List[Document]:
        """
        Rerank retrieved documents using the original user query.

        Parameters
        ----------
        query : str
            Original user query.

        docs : List[Document]
            Candidate documents retrieved upstream.

        Returns
        -------
        List[Document]
            Top reranked documents after light deduplication.
        """
        # Return early if there are no candidate documents.
        if not docs:
            return []

        # Build query-document pairs for cross-encoder scoring.
        pairs = [(query, doc.page_content) for doc in docs]

        # Predict a relevance score for each query-document pair.
        scores = self.model.predict(pairs)

        # Sort documents by descending relevance score.
        ranked = sorted(
            zip(docs, scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )

        # Apply light deduplication to improve diversity in the final set.
        selected: List[Document] = []
        seen_content = set()

        for doc, _score in ranked:
            # Build a normalized content fingerprint for duplicate detection.
            content_key = normalize(doc.page_content)

            # Skip documents with duplicate or near-duplicate content.
            if content_key in seen_content:
                continue

            # Keep the first occurrence of each unique content block.
            seen_content.add(content_key)
            selected.append(doc)

            # Stop when the configured top-N limit is reached.
            if len(selected) >= self.top_n:
                break

        return selected