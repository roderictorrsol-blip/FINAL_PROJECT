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
        """
        Format citations for internal prompt use only.
        The final UI will display sources separately.
        """
        if not citations:
            return "No citation URLs available."

        return "\n".join(f"- {url}" for url in citations)

    def _build_prompt(self, question: str, context: str, citations: List[str]) -> str:
        formatted_citations = self._format_citations(citations)

        return (
            "You are an expert assistant on World War II.\n"
            "Answer the user's question using ONLY the provided context.\n"
            "Do not add facts that are not supported by the context.\n"
            "If the context is insufficient, say so clearly.\n"
            "You may make careful, limited inferences only when they are strongly supported by the context.\n"
            "Prefer a clear and direct historical explanation over long quotation.\n"
            "If a short quote from the context is genuinely helpful, you may include it.\n"
            "Do NOT include a list of URLs, a 'Sources' section, or a 'Citations' section in the answer.\n"
            "The application will display the sources separately.\n\n"
            f"QUESTION:\n{question}\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"AVAILABLE SOURCE URLS:\n{formatted_citations}\n"
        )

    def answer(self, question: str, context: str, citations: List[str]) -> str:
        formatted_citations = self._format_citations(citations)

        response = self.client.responses.create(
            model=self.chat_model,
            instructions=(
                "You are an expert assistant on World War II. "
                "Answer using only the provided context. "
                "Do not add unsupported facts. "
                "If the context is insufficient, say so clearly. "
                "Do not include a Sources or Citations section."
            ),
            input=(
                f"QUESTION:\n{question}\n\n"
                f"CONTEXT:\n{context}\n\n"
                f"AVAILABLE SOURCE URLS:\n{formatted_citations}"
            ),
        )
        return response.output_text.strip()