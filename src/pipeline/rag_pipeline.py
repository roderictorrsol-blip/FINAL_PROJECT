"""
rag_pipeline.py

Reusable RAG pipeline for transcript-based QA.

Pipeline steps:
1. Query rewriting
2. Retrieval
3. Context building
4. Citation formatting
5. Answer generation

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
        result = pipeline.run("¿Qué dice sobre el ojo?")
        print(result["answer"])
    """

    def __init__(
        self,
        num_rewrites: int = 4,
        top_k_per_query: int = 5,
        max_final_docs: int = 5,
        max_citations: int = 5,
    ) -> None:
        self.query_rewriter = QueryRewriter(num_rewrites=num_rewrites)
        self.retriever = RetrieverAgent(top_k=top_k_per_query)
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
                "docs": [],
                "context": "",
                "citations": [],
                "answer": "",
            }

        # 1. Rewrite query
        queries: List[str] = self.query_rewriter.rewrite(question)

        # 2. Retrieve docs (prioritize original question)
        raw_docs = self.retriever.retrieve(question, queries)

        # 3. Build context
        context_data = self.context_builder.build(raw_docs)
        docs = context_data["docs"]
        context = context_data["context"]
        citations_raw = context_data["citations"]

        # 4. Format citations
        citations = self.citation_formatter.as_list(citations_raw)

        # 5. Generate answer
        answer = self.answer_agent.answer(
            question=question,
            context=context,
            citations=citations,
        )

        return {
            "question": question,
            "queries": queries,
            "docs": docs,
            "context": context,
            "citations": citations,
            "answer": answer,
        }