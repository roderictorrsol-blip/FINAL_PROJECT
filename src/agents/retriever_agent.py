"""
retriever_agent.py

Retriever agent for RAG.

Responsibilities:
- Receive the original question and rewritten queries
- Query the FAISS vectorstore
- Prioritize results from the original question
- Add rewritten-query results as secondary expansion
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


DEFAULT_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_STORE_DIR = "data/vectorstore/faiss_store_openai"
DEFAULT_TOP_K = 10


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


class RetrieverAgent:
    """
    FAISS retriever agent.

    Strategy:
    - First retrieve with the original user question
    - Then expand with rewritten queries
    - Keep ordering stable so original-question results are prioritized
    """

    def __init__(
        self,
        store_dir: str = DEFAULT_STORE_DIR,
        embed_model: str = DEFAULT_EMBED_MODEL,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        base = project_root()
        store_path = base / store_dir

        if not store_path.exists():
            raise FileNotFoundError(
                f"Vectorstore not found: {store_path}. "
                "Run src/03_build_vectorstore.py first."
            )

        self.embeddings = OpenAIEmbeddings(model=embed_model)
        self.vectorstore = FAISS.load_local(
            str(store_path),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": top_k}
        )

    def retrieve(self, original_question: str, queries: List[str]) -> List[Document]:
        """
        Retrieve documents prioritizing the original question.

        Order:
        1. Results from the original question
        2. Results from rewritten queries
        """
        docs: List[Document] = []

        # 1) Primary retrieval: original question first
        docs.extend(self.retriever.invoke(original_question))

        # 2) Secondary retrieval: rewritten queries
        for q in queries:
            if q.strip() == original_question.strip():
                continue
            docs.extend(self.retriever.invoke(q))

        return docs