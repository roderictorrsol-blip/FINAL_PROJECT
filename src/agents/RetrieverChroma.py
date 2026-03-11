"""
retriever_chroma.py

Vector-based retriever using a Chroma persistent database.

Responsibilities:
- Connect to the Chroma vector store
- Run semantic similarity search
- Return LangChain Document objects for downstream agents
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


DEFAULT_COLLECTION = "wwii_chunks"
DEFAULT_CHROMA_DIR = "data/chroma"
DEFAULT_EMBED_MODEL = "text-embedding-3-large"
DEFAULT_TOP_K = 8


def project_root() -> Path:
    """
    Resolve the project root directory.
    """
    return Path(__file__).resolve().parents[2]


class RetrieverChroma:
    """
    Semantic retriever backed by a persistent Chroma vector store.

    This retriever performs embedding-based similarity search
    over transcript chunks stored in Chroma.
    """

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION,
        persist_directory: str = DEFAULT_CHROMA_DIR,
        embedding_model: str = DEFAULT_EMBED_MODEL,
    ) -> None:

        # Load environment variables from project .env
        base = project_root()
        load_dotenv(base / ".env")

        # Ensure OpenAI API key exists
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("Missing OPENAI_API_KEY in FINAL_PROJECT/.env")

        # Create embedding model
        self.embedding = OpenAIEmbeddings(model=embedding_model)

        # Initialize the Chroma vector store
        self.vectorstore = Chroma(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_function=self.embedding,
        )

    def retrieve(self, query: str, k: int = DEFAULT_TOP_K) -> List[Document]:
        """
        Retrieve documents using semantic similarity search.

        Parameters
        ----------
        query : str
            The search query.

        k : int
            Number of documents to return.

        Returns
        -------
        List[Document]
            Retrieved documents ordered by similarity.
        """
        return self.vectorstore.similarity_search(query, k=k)