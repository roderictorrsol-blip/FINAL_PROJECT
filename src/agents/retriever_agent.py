"""
retriever_agent.py

Retriever agent for RAG.

Responsibilities:
- Receive the original question and rewritten queries
- Query the selected retrieval backend
- Support:
    - FAISS
    - Chroma
    - Hybrid (FAISS + BM25)
- Prioritize results from the original question
- Add rewritten-query results as secondary expansion
- Deduplicate documents before passing them downstream

Architecture intent:
- Chroma = persistent source of truth
- FAISS = fast snapshot index
- BM25 = lexical complement
- Reranker = final selector (outside this agent)
"""

# Enable postponed evaluation of type annotations.
from __future__ import annotations

# Standard library imports.
import logging
import os
from pathlib import Path
from typing import List

# Third-party imports.
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Project imports.
from src.agents.retriever_bm25 import RetrieverBM25

# Load environment variables from the project .env file.
load_dotenv()

# Create a module-level logger for retrieval diagnostics.
logger = logging.getLogger(__name__)

# Retrieval backend selected through environment configuration.
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "faiss").lower()

# Default embedding model used by semantic retrievers.
DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")

# Storage locations for the supported vector backends.
DEFAULT_FAISS_STORE_DIR = "data/vectorstore/faiss_store_openai"
DEFAULT_CHROMA_DIR = "data/chroma"
DEFAULT_CHROMA_COLLECTION = "wwii_chunks"

# Retrieval depth configuration.
DEFAULT_VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "8"))
DEFAULT_BM25_TOP_K = int(os.getenv("BM25_TOP_K", "5"))
DEFAULT_FINAL_RETRIEVAL_K = int(os.getenv("FINAL_RETRIEVAL_K", "10"))


def project_root() -> Path:
    """
    Resolve the project root directory.

    This assumes the file lives under:
        project_root/src/agents/retriever_agent.py
    """
    return Path(__file__).resolve().parents[2]


def normalize_content(text: str, limit: int = 500) -> str:
    """
    Normalize content for stronger duplicate detection.

    This helper:
    - replaces line breaks with spaces
    - collapses repeated whitespace
    - lowercases the text
    - truncates the result to a fixed limit
    """
    text = (text or "").replace("\n", " ").strip()
    text = " ".join(text.split()).lower()
    return text[:limit]


class RetrieverAgent:
    """
    Retriever agent supporting:
    - FAISS
    - Chroma
    - Hybrid (FAISS + BM25)

    Strategy:
    - First retrieve with the original user question
    - Then expand with rewritten queries
    - Keep ordering stable so original-question results are prioritized
    """

    def __init__(
        self,
        embed_model: str = DEFAULT_EMBED_MODEL,
        vector_top_k: int = DEFAULT_VECTOR_TOP_K,
        bm25_top_k: int = DEFAULT_BM25_TOP_K,
        final_retrieval_k: int = DEFAULT_FINAL_RETRIEVAL_K,
    ) -> None:
        """
        Initialize the retriever agent and the configured retrieval backend.

        Parameters
        ----------
        embed_model : str
            Embedding model used by semantic retrievers.

        vector_top_k : int
            Number of semantic documents returned per query.

        bm25_top_k : int
            Number of lexical BM25 documents returned per query.

        final_retrieval_k : int
            Maximum number of documents passed downstream after merge and deduplication.
        """
        # Store retrieval configuration.
        self.embed_model = embed_model
        self.vector_top_k = vector_top_k
        self.bm25_top_k = bm25_top_k
        self.final_retrieval_k = final_retrieval_k

        # Create the embedding model used by FAISS and Chroma.
        self.embeddings = OpenAIEmbeddings(model=embed_model)

        # Resolve the project root to build backend paths.
        base = project_root()

        # Initialize retriever placeholders.
        self.vector_retriever = None
        self.bm25_retriever = None

        logger.info("Initializing RetrieverAgent with backend='%s'", VECTOR_BACKEND)
        logger.info("Embedding model='%s'", self.embed_model)
        logger.info("vector_top_k=%d", self.vector_top_k)
        logger.info("bm25_top_k=%d", self.bm25_top_k)
        logger.info("final_retrieval_k=%d", self.final_retrieval_k)

        if VECTOR_BACKEND == "faiss":
            # Pure semantic retrieval using FAISS.
            self.vector_retriever = self._build_faiss_retriever(base)

        elif VECTOR_BACKEND == "chroma":
            # Pure semantic retrieval using the persistent Chroma store.
            self.vector_retriever = self._build_chroma_retriever(base)

        elif VECTOR_BACKEND == "hybrid":
            # Hybrid retrieval uses FAISS as the fast semantic index
            # and BM25 as the lexical complement.
            self.vector_retriever = self._build_faiss_retriever(base)
            self.bm25_retriever = RetrieverBM25(
                top_k=self.bm25_top_k,
            )

        else:
            raise ValueError(
                f"Unsupported VECTOR_BACKEND='{VECTOR_BACKEND}'. "
                "Use 'faiss', 'chroma', or 'hybrid'."
            )

    def _build_faiss_retriever(self, base: Path):
        """
        Load the FAISS vector store and return a retriever wrapper.
        """
        # Resolve the FAISS index path.
        store_path = base / DEFAULT_FAISS_STORE_DIR

        # Ensure the FAISS store exists before loading it.
        if not store_path.exists():
            raise FileNotFoundError(
                f"FAISS store not found: {store_path}. "
                "Run the FAISS build script first."
            )

        # Load the local FAISS vector store.
        vectorstore = FAISS.load_local(
            str(store_path),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

        logger.info("Loaded FAISS store from %s", store_path)

        # Return a LangChain retriever wrapper around the vector store.
        return vectorstore.as_retriever(
            search_kwargs={"k": self.vector_top_k}
        )

    def _build_chroma_retriever(self, base: Path):
        """
        Load the persistent Chroma store and return a retriever wrapper.
        """
        try:
            from langchain_chroma import Chroma
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "langchain_chroma is not installed. "
                "Install it with: python -m pip install langchain-chroma chromadb"
            ) from e

        # Resolve the Chroma persistence directory.
        chroma_path = base / DEFAULT_CHROMA_DIR

        # Ensure the Chroma store exists before loading it.
        if not chroma_path.exists():
            raise FileNotFoundError(
                f"Chroma store not found: {chroma_path}. "
                "Run the Chroma build script first."
            )

        # Load the persistent Chroma vector store.
        vectorstore = Chroma(
            collection_name=DEFAULT_CHROMA_COLLECTION,
            persist_directory=str(chroma_path),
            embedding_function=self.embeddings,
        )

        logger.info("Loaded Chroma store from %s", chroma_path)

        # Return a LangChain retriever wrapper around the vector store.
        return vectorstore.as_retriever(
            search_kwargs={"k": self.vector_top_k}
        )

    def _deduplicate(self, docs: List[Document]) -> List[Document]:
        """
        Remove duplicate documents while preserving order.

        Duplicate detection priority:
        1. chunk_id
        2. normalized content fingerprint
        """
        unique_docs: List[Document] = []
        seen_chunk_ids = set()
        seen_content = set()

        for doc in docs:
            # Read metadata safely from the current document.
            meta = getattr(doc, "metadata", {}) or {}
            chunk_id = meta.get("chunk_id")

            # Build a normalized fingerprint from the page content.
            content_key = normalize_content(doc.page_content)

            if chunk_id:
                # Prefer chunk_id when available, but also guard against repeated content.
                if chunk_id in seen_chunk_ids or content_key in seen_content:
                    continue
                seen_chunk_ids.add(chunk_id)
                seen_content.add(content_key)
            else:
                # If no chunk_id exists, deduplicate only by normalized content.
                if content_key in seen_content:
                    continue
                seen_content.add(content_key)

            unique_docs.append(doc)

        return unique_docs

    def _retrieve_vector(self, query: str) -> List[Document]:
        """
        Retrieve documents using the semantic retriever.
        """
        docs = self.vector_retriever.invoke(query)
        logger.info("[Retriever] vector_docs=%d", len(docs))
        return docs

    def _retrieve_bm25(self, query: str) -> List[Document]:
        """
        Retrieve documents using the BM25 lexical retriever.
        """
        if self.bm25_retriever is None:
            return []

        docs = self.bm25_retriever.retrieve(query)
        logger.info("[Retriever] bm25_docs=%d", len(docs))
        return docs

    def _retrieve_once(self, query: str) -> List[Document]:
        """
        Retrieve documents for a single query according to the selected backend.
        """
        logger.info("[Retriever] query='%s' backend=%s", query, VECTOR_BACKEND)

        if VECTOR_BACKEND in {"faiss", "chroma"}:
            return self._retrieve_vector(query)

        if VECTOR_BACKEND == "hybrid":
            docs: List[Document] = []

            # Retrieve semantic candidates from FAISS.
            vector_docs = self._retrieve_vector(query)

            # Retrieve lexical candidates from BM25.
            bm25_docs = self._retrieve_bm25(query)

            # Keep vector results first, then append lexical results.
            docs.extend(vector_docs)
            docs.extend(bm25_docs)

            # Deduplicate the merged result set before returning it.
            merged_docs = len(docs)
            unique_docs = self._deduplicate(docs)

            logger.info("[Retriever] merged_docs=%d", merged_docs)
            logger.info("[Retriever] unique_docs=%d", len(unique_docs))

            return unique_docs

        return []

    def retrieve(self, original_question: str, queries: List[str]) -> List[Document]:
        """
        Retrieve documents prioritizing the original question.

        Order:
        1. Results from the original question
        2. Results from rewritten queries
        """
        docs: List[Document] = []

        # Primary retrieval: original question first.
        docs.extend(self._retrieve_once(original_question))

        # Secondary retrieval: rewritten queries.
        for q in queries:
            if q.strip() == original_question.strip():
                continue
            docs.extend(self._retrieve_once(q))

        # Apply a final deduplication pass after merging all candidate results.
        final_docs = self._deduplicate(docs)

        # Limit the number of documents passed to the reranker.
        final_docs = final_docs[: self.final_retrieval_k]

        logger.info("[Retriever] final_unique_docs=%d", len(final_docs))

        return final_docs