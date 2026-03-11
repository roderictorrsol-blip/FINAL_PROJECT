"""
answer_agent.py

Answer generation agent for RAG.

Responsibilities:
- Receive question, context, and citations
- Call the LLM
- Return the final grounded answer
"""

# Enable postponed evaluation of type annotations.
from __future__ import annotations

# Standard library imports.
import os
from pathlib import Path
from typing import List

# Load environment variables.
from dotenv import load_dotenv

# OpenAI client used to generate answers.
from openai import OpenAI


# Default chat model used for answer generation.
DEFAULT_CHAT_MODEL = "gpt-4o-mini"


def project_root() -> Path:
    """
    Resolve the project root directory.
    """
    return Path(__file__).resolve().parents[2]


class AnswerAgent:
    """
    Answer generation agent for transcript-based RAG.

    This agent receives:
    - the user question
    - the retrieved context
    - citation URLs

    It then calls the LLM to generate a grounded answer.

    Example:
        agent = AnswerAgent()

        answer = agent.answer(
            question="What is RAG?",
            context="...",
            citations=["https://..."]
        )
    """

    def __init__(self, chat_model: str = DEFAULT_CHAT_MODEL) -> None:
        """
        Initialize the answer agent and OpenAI client.
        """
        base = project_root()

        # Load environment variables from project .env
        load_dotenv(base / ".env")

        # Ensure API key exists
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("Missing OPENAI_API_KEY in FINAL_PROJECT/.env")

        # Store model configuration
        self.chat_model = chat_model

        # Create OpenAI client
        self.client = OpenAI()

    @staticmethod
    def _format_citations(citations: List[str]) -> str:
        """
        Format citation URLs for prompt inclusion.

        These citations are used internally by the model
        but the UI will display them separately.
        """
        if not citations:
            return "No citation URLs available."

        return "\n".join(f"- {url}" for url in citations)

    def _build_prompt(self, question: str, context: str, citations: List[str]) -> str:
        """
        Build the prompt sent to the LLM.
        """
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
        """
        Generate a grounded answer using the LLM.
        """

        # Build the final prompt
        prompt = self._build_prompt(question, context, citations)

        # Call the OpenAI Responses API
        response = self.client.responses.create(
            model=self.chat_model,
            instructions=(
                "You are an expert assistant on World War II. "
                "Answer using only the provided context. "
                "Do not add unsupported facts. "
                "If the context is insufficient, say so clearly. "
                "Do not include a Sources or Citations section."
            ),
            input=prompt,
        )

        # Extract the text output produced by the model
        return response.output_text.strip()