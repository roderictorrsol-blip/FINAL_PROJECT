"""
rag_pipeline.py

Reusable RAG pipeline for transcript-based QA.

Pipeline steps:
1. Query rewriting
2. Retrieval
3. Reranking
4. Context building
5. Citation formatting
6. Answer generation

Main entry point:
    RAGPipeline.run(question)
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.agents.query_rewriter import QueryRewriter
from src.agents.retriever_agent import RetrieverAgent
from src.agents.reranker_agent import RerankerAgent
from src.agents.context_builder import ContextBuilder
from src.agents.citation_formatter import CitationFormatter
from src.agents.answer_agent import AnswerAgent


class RAGPipeline:
    """
    Reusable RAG pipeline.

    Example:
        pipeline = RAGPipeline()
        result = pipeline.run("¿Qué ocurrió en la invasión de Polonia?")
        print(result["answer"])
    """

    def __init__(
        self,
        num_rewrites: int = 4,
        max_final_docs: int = 5,
        max_citations: int = 5,
    ) -> None:
        self.query_rewriter = QueryRewriter(num_rewrites=num_rewrites)
        self.retriever = RetrieverAgent()
        self.reranker = RerankerAgent(top_n=max_final_docs)
        self.context_builder = ContextBuilder(max_docs=max_final_docs)
        self.citation_formatter = CitationFormatter(max_citations=max_citations)
        self.answer_agent = AnswerAgent()

    def run(self, question: str) -> Dict[str, Any]:
        """
        Run the full RAG pipeline.

        Returns a structured dictionary with:
        - question
        - queries
        - raw_docs
        - docs
        - context
        - citations
        - answer
        """
        question = question.strip()

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

        # 1. Rewrite query
        queries: List[str] = self.query_rewriter.rewrite(question)

        # 2. Retrieve docs (prioritize original question)
        raw_docs = self.retriever.retrieve(question, queries)

        # 3. Rerank retrieved docs
        reranked_docs = self.reranker.rerank(question, raw_docs)

        # 4. Build context from reranked docs
        context_data = self.context_builder.build(reranked_docs)
        docs = context_data["docs"]
        context = context_data["context"]
        citations_raw = context_data["citations"]

        # 5. Format citations
        citations = self.citation_formatter.as_list(citations_raw)

        # 6. Generate answer
        answer = self.answer_agent.answer(
            question=question,
            context=context,
            citations=citations,
        )

        return {
            "question": question,
            "queries": queries,
            "raw_docs": raw_docs,
            "docs": docs,
            "context": context,
            "citations": citations,
            "answer": answer,
        }