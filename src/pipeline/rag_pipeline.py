"""
rag_pipeline.py

Reusable RAG pipeline for transcript-based question answering.

Pipeline steps:
1. Query rewriting
2. Document retrieval
3. Document reranking
4. Context building
5. Citation formatting
6. Answer generation

Main entry point:
    RAGPipeline.run(question)
"""

# Enable postponed evaluation of type annotations (Python 3.7+ compatibility).
from __future__ import annotations

# Typing utilities used for clearer function signatures.
from typing import Any, Dict, List

# Import the agents responsible for each step of the RAG pipeline.
from src.agents.query_rewriter import QueryRewriter
from src.agents.retriever_agent import RetrieverAgent
from src.agents.reranker_agent import RerankerAgent
from src.agents.context_builder import ContextBuilder
from src.agents.citation_formatter import CitationFormatter
from src.agents.answer_agent import AnswerAgent


class RAGPipeline:
    """
    Reusable Retrieval-Augmented Generation pipeline.

    This class orchestrates the full QA flow:
    - expanding the user query
    - retrieving relevant documents
    - reranking the documents
    - building a context window
    - formatting citations
    - generating the final answer

    The retrieval layer is configurable and can use:
    - FAISS
    - Chroma
    - hybrid retrieval (FAISS + BM25)

    Example
    -------
    pipeline = RAGPipeline()
    result = pipeline.run("What happened during the invasion of Poland?")
    print(result["answer"])
    """

    def __init__(
        self,
        num_rewrites: int = 4,
        max_final_docs: int = 5,
        max_citations: int = 5,
    ) -> None:
        """
        Initialize the RAG pipeline and its internal agents.

        Parameters
        ----------
        num_rewrites : int
            Number of alternative queries generated from the original question.

        max_final_docs : int
            Maximum number of documents kept after reranking.

        max_citations : int
            Maximum number of citations included in the final answer.
        """

        # Agent responsible for generating multiple search queries.
        self.query_rewriter = QueryRewriter(num_rewrites=num_rewrites)

        # Agent responsible for retrieving documents from the configured backend
        # (FAISS, Chroma, or hybrid retrieval).
        self.retriever = RetrieverAgent()

        # Agent that reranks retrieved documents based on relevance.
        self.reranker = RerankerAgent(top_n=max_final_docs)

        # Agent that builds the context window sent to the LLM.
        self.context_builder = ContextBuilder(max_docs=max_final_docs)

        # Agent that formats citations for display.
        self.citation_formatter = CitationFormatter(max_citations=max_citations)

        # Agent responsible for generating the final LLM answer.
        self.answer_agent = AnswerAgent()

    def run(self, question: str) -> Dict[str, Any]:
        """
        Execute the full RAG pipeline for a given question.

        Parameters
        ----------
        question : str
            The user's input question.

        Returns
        -------
        Dict[str, Any]
            Structured output containing:
            - question
            - queries
            - raw_docs
            - docs
            - context
            - citations
            - answer
        """

        # Remove leading/trailing whitespace from the question.
        question = question.strip()

        # Handle empty input gracefully.
        if not question:
            return {
                "question": "",
                "queries": [],
                "raw_docs": [],
                "docs": [],
                "context": "",
                "citations": [],
                "answer": "",
            }

        # ---------------------------------------------------
        # Step 1 — Query rewriting
        # Generate alternative search queries to improve recall.
        # ---------------------------------------------------
        queries: List[str] = self.query_rewriter.rewrite(question)

        # ---------------------------------------------------
        # Step 2 — Retrieval
        # Retrieve candidate documents using the configured retrieval strategy.
        # This may be FAISS, Chroma, or hybrid retrieval depending on settings.
        # The original question is prioritized.
        # ---------------------------------------------------
        raw_docs = self.retriever.retrieve(question, queries)

        # ---------------------------------------------------
        # Step 3 — Reranking
        # Reorder retrieved documents based on semantic relevance.
        # ---------------------------------------------------
        reranked_docs = self.reranker.rerank(question, raw_docs)

        # ---------------------------------------------------
        # Step 4 — Context building
        # Construct the text context passed to the LLM.
        # ---------------------------------------------------
        context_data = self.context_builder.build(reranked_docs)

        docs = context_data["docs"]
        context = context_data["context"]
        citations_raw = context_data["citations"]

        # ---------------------------------------------------
        # Step 5 — Citation formatting
        # Convert raw citation metadata into user-readable format.
        # ---------------------------------------------------
        citations = self.citation_formatter.as_list(citations_raw)

        # ---------------------------------------------------
        # Step 6 — Answer generation
        # Use the LLM to generate the final answer.
        # ---------------------------------------------------
        answer = self.answer_agent.answer(
            question=question,
            context=context,
            citations=citations,
        )

        # Return structured output for downstream consumers.
        return {
            "question": question,
            "queries": queries,
            "raw_docs": raw_docs,
            "docs": docs,
            "context": context,
            "citations": citations,
            "answer": answer,
        }