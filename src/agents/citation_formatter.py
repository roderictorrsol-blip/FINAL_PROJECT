"""
citation_formatter.py

Citation formatting agent for RAG.

Responsibilities:
- Normalize citation URLs
- Remove duplicates while preserving order
- Provide output as list or formatted text
"""

from __future__ import annotations

from typing import List


DEFAULT_MAX_CITATIONS = 5


class CitationFormatter:
    """
    Citation formatter for transcript-based RAG.

    Example:
        formatter = CitationFormatter()
        clean = formatter.as_list(citations)
        text = formatter.as_text(citations)
    """

    def __init__(self, max_citations: int = DEFAULT_MAX_CITATIONS) -> None:
        self.max_citations = max_citations

    @staticmethod
    def _normalize(url: str) -> str:
        """
        Normalize a citation URL.
        """
        return (url or "").strip()

    def as_list(self, citations: List[str]) -> List[str]:
        """
        Return a clean deduplicated list of citations.
        """
        seen = set()
        out: List[str] = []

        for url in citations:
            clean = self._normalize(url)
            if not clean:
                continue
            if clean in seen:
                continue

            seen.add(clean)
            out.append(clean)

            if len(out) >= self.max_citations:
                break

        return out

    def as_text(self, citations: List[str]) -> str:
        """
        Return citations as newline-separated text.
        """
        return "\n".join(self.as_list(citations))