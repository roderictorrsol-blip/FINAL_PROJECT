"""
citation_formatter.py

Citation formatting agent for RAG.

Responsibilities:
- Normalize citation URLs
- Remove duplicates while preserving order
- Provide output as a list or formatted text
"""

# Enable postponed evaluation of type annotations.
from __future__ import annotations

# Typing utilities for citation collections.
from typing import List


# Default maximum number of citations included in the final output.
DEFAULT_MAX_CITATIONS = 5


class CitationFormatter:
    """
    Citation formatter for transcript-based RAG.

    This agent standardizes the final citation output by:
    - normalizing URLs
    - removing duplicates
    - limiting the maximum number of citations
    - returning citations as a list or as plain text

    Example:
        formatter = CitationFormatter()
        clean = formatter.as_list(citations)
        text = formatter.as_text(citations)
    """

    def __init__(self, max_citations: int = DEFAULT_MAX_CITATIONS) -> None:
        """
        Initialize the citation formatter.

        Parameters
        ----------
        max_citations : int
            Maximum number of unique citations returned.
        """
        self.max_citations = max_citations

    @staticmethod
    def _normalize(url: str) -> str:
        """
        Normalize a citation URL.

        Current normalization is intentionally minimal:
        - convert None-like inputs to an empty string
        - strip leading and trailing whitespace
        """
        return (url or "").strip()

    def as_list(self, citations: List[str]) -> List[str]:
        """
        Return a clean, deduplicated list of citations.

        Behavior:
        - removes empty values
        - removes duplicates
        - preserves original order
        - limits output to max_citations
        """
        seen = set()
        out: List[str] = []

        for url in citations:
            # Normalize the raw URL string.
            clean = self._normalize(url)

            # Skip empty citations.
            if not clean:
                continue

            # Skip repeated citations while preserving the first occurrence.
            if clean in seen:
                continue

            # Keep the citation.
            seen.add(clean)
            out.append(clean)

            # Stop when the configured maximum is reached.
            if len(out) >= self.max_citations:
                break

        return out

    def as_text(self, citations: List[str]) -> str:
        """
        Return citations as newline-separated text.

        This is useful for prompts, logs, or simple plain-text rendering.
        """
        return "\n".join(self.as_list(citations))