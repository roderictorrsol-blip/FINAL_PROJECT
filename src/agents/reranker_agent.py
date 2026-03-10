from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RerankerAgent:
    """
    Reranks retrieved documents using a cross-encoder model.
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

        return [doc for doc, _ in ranked[: self.top_n]]