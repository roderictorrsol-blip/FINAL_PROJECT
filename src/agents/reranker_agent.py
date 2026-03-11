from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def normalize(text: str, limit: int = 400) -> str:
    text = (text or "").replace("\n", " ").strip()
    text = " ".join(text.split()).lower()
    return text[:limit]


class RerankerAgent:
    """
    Reranks retrieved documents using a cross-encoder model
    and applies light deduplication to improve diversity.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANK_MODEL,
        top_n: int = 5,
    ) -> None:
        self.model = CrossEncoder(model_name)
        self.top_n = top_n

    def rerank(self, query: str, docs: List[Document]) -> List[Document]:
        if not docs:
            return []

        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(docs, scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )

        # diversify results
        selected: List[Document] = []
        seen_content = set()

        for doc, score in ranked:
            content_key = normalize(doc.page_content)

            if content_key in seen_content:
                continue

            seen_content.add(content_key)
            selected.append(doc)

            if len(selected) >= self.top_n:
                break

        return selected