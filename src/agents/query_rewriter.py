"""
query_rewriter.py

Query rewriter agent for RAG.

Responsibilities:
- Receive the original user question
- Generate multiple rewritten search queries
- Preserve the original question as fallback
- Return a clean deduplicated list of queries
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from openai import OpenAI


DEFAULT_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_NUM_REWRITES = 5


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


class QueryRewriter:
    """
    Multi-query rewriter for transcript retrieval.

    Example:
        rewriter = QueryRewriter()
        queries = rewriter.rewrite("¿Qué dice sobre el ojo?")
    """

    def __init__(
        self,
        chat_model: str = DEFAULT_CHAT_MODEL,
        num_rewrites: int = DEFAULT_NUM_REWRITES,
    ) -> None:
        base = project_root()
        load_dotenv(base / ".env")

        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("Missing OPENAI_API_KEY in FINAL_PROJECT/.env")

        self.chat_model = chat_model
        self.num_rewrites = num_rewrites
        self.client = OpenAI()

    @staticmethod
    def _parse_queries(text: str, original_question: str, max_rewrites: int) -> List[str]:
        """
        Parse model output into a deduplicated list of queries.
        Always keeps the original question as first fallback query.
        """
        queries = [original_question.strip()]

        for line in text.splitlines():
            q = line.strip().lstrip("-•1234567890. ").strip()
            if q and q not in queries:
                queries.append(q)

        return queries[: max_rewrites + 1]

    def _build_prompt(self, question: str) -> str:
        return (
            "Generate multiple search queries to improve retrieval from a "
            "transcript-based knowledge base.\n"
            f"Return exactly {self.num_rewrites} short query variations, one per line.\n"
            "Your goal is retrieval effectiveness, not stylistic paraphrasing.\n"
            "Rules:\n"
            "- Preserve the original meaning.\n"
            "- Do NOT generate only synonyms or surface paraphrases.\n"
            "- Make the query more specific and retrieval-friendly.\n"
            "- When the question is vague, infer the most likely topic and expand it.\n"
            "- When possible, include relevant entities, dates, places, events, historical actors, or named concepts that may appear in transcripts.\n"
            "- Prefer concrete search terms over abstract wording.\n"
            "- Prefer queries that are likely to appear verbatim in transcripts or documentaries.\n"
            "- If the topic is historical, include likely event names, countries, leaders, battles, dates, or turning points.\n"
            "- If the user asks in Spanish, return queries in Spanish.\n"
            "- If the user asks in English, return queries in English.\n"
            "- Do not explain anything.\n"
            "- Output only the queries.\n\n"
            f"QUESTION:\n{question}"
        )

    def rewrite(self, question: str) -> List[str]:
        """
        Rewrite a user question into multiple retrieval-friendly queries.
        """
        question = question.strip()
        if not question:
            return []

        response = self.client.responses.create(
            model=self.chat_model,
            input=self._build_prompt(question),
        )

        rewrite_text = response.output_text.strip()
        return self._parse_queries(rewrite_text, question, self.num_rewrites)