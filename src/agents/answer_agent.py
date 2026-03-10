"""
answer_agent.py

Answer generation agent for RAG.

Responsibilities:
- Receive question, context, and citations
- Call the LLM
- Return the final grounded answer
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from openai import OpenAI


DEFAULT_CHAT_MODEL = "gpt-4o-mini"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


class AnswerAgent:
    """
    Answer generation agent for transcript-based RAG.

    Example:
        agent = AnswerAgent()
        answer = agent.answer(
            question="What is RAG?",
            context="...",
            citations=["https://..."]
        )
    """

    def __init__(self, chat_model: str = DEFAULT_CHAT_MODEL) -> None:
        base = project_root()
        load_dotenv(base / ".env")

        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("Missing OPENAI_API_KEY in FINAL_PROJECT/.env")

        self.chat_model = chat_model
        self.client = OpenAI()

    @staticmethod
    def _format_citations(citations: List[str]) -> str:
        return "\n".join(citations)

    def _build_prompt(self, question: str, context: str, citations: List[str]) -> str:
        return (
        "You´re an expert in World War II.\n"
        "Answer using ONLY the provided CONTEXT.\n"
        "Use the context to infer the answer when possible.\n"
        "If the answer is partially present, explain it based on the context.\n"
        "If the answer is partially present, explain it based on the context.\n"
        "Include 1-3 short quotes from the CONTEXT in quotation marks when helpful.\n"
        "Do NOT include a list of URLs or a 'Citations' section in your answer.\n"
        "The application will display the sources separately.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT:\n{context}\n"
    )

    def answer(self, question: str, context: str, citations: List[str]) -> str:
        """
        Generate the final grounded answer.
        """
        response = self.client.responses.create(
            model=self.chat_model,
            input=self._build_prompt(question, context, citations),
        )
        return response.output_text.strip()