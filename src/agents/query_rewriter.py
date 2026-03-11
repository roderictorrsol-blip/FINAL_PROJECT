"""
query_rewriter.py

Query rewriter agent for RAG.

Responsibilities:
- Receive the original user question
- Generate multiple rewritten search queries
- Preserve the original question as a fallback
- Return a clean, deduplicated list of queries for retrieval
"""

# Enable postponed evaluation of type annotations.
from __future__ import annotations

# Standard library imports.
import os
from pathlib import Path
from typing import List

# Load environment variables from the project's .env file.
from dotenv import load_dotenv

# OpenAI client used to generate rewritten retrieval queries.
from openai import OpenAI


# Default chat model used for query rewriting.
DEFAULT_CHAT_MODEL = "gpt-4o-mini"

# Default number of rewritten queries requested from the model.
DEFAULT_NUM_REWRITES = 5


def project_root() -> Path:
    """
    Resolve the project root directory.

    This assumes the file structure:
        project_root/
            src/
                agents/
                    query_rewriter.py
    """
    return Path(__file__).resolve().parents[2]


class QueryRewriter:
    """
    Multi-query rewriter for transcript retrieval.

    This agent expands a user's question into multiple retrieval-friendly
    queries in order to improve recall during document search.

    Example:
        rewriter = QueryRewriter()
        queries = rewriter.rewrite("¿Qué dice sobre el ojo?")
    """

    def __init__(
        self,
        chat_model: str = DEFAULT_CHAT_MODEL,
        num_rewrites: int = DEFAULT_NUM_REWRITES,
    ) -> None:
        """
        Initialize the query rewriter.

        Parameters
        ----------
        chat_model : str
            OpenAI chat model used to generate rewritten queries.

        num_rewrites : int
            Number of alternative retrieval queries to request.
        """

        # Resolve the project root so the local .env file can be loaded.
        base = project_root()

        # Load environment variables from the project-level .env file.
        load_dotenv(base / ".env")

        # Ensure the OpenAI API key is available before creating the client.
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("Missing OPENAI_API_KEY in FINAL_PROJECT/.env")

        # Store configuration values.
        self.chat_model = chat_model
        self.num_rewrites = num_rewrites

        # Create the OpenAI client instance.
        self.client = OpenAI()

    @staticmethod
    def _parse_queries(
        text: str,
        original_question: str,
        max_rewrites: int,
    ) -> List[str]:
        """
        Parse the model output into a deduplicated list of retrieval queries.

        Behavior:
        - Always keeps the original question as the first fallback query
        - Removes simple numbering and bullet prefixes
        - Deduplicates repeated lines
        - Returns at most (max_rewrites + 1) queries because the original
          question is included in the final list
        """
        # Start with the original question so retrieval always has a safe fallback.
        queries = [original_question.strip()]

        # Process the model output line by line.
        for line in text.splitlines():
            # Remove whitespace, common bullets, and leading numbering.
            q = line.strip().lstrip("-•1234567890. ").strip()

            # Keep only non-empty and non-duplicate queries.
            if q and q not in queries:
                queries.append(q)

        # Keep the original question plus up to max_rewrites generated queries.
        return queries[: max_rewrites + 1]

    def _build_prompt(self, question: str) -> str:
        """
        Build the prompt used to generate retrieval-oriented query rewrites.
        """
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

        Parameters
        ----------
        question : str
            The original user question.

        Returns
        -------
        List[str]
            A deduplicated list containing the original question and the
            generated retrieval rewrites.
        """
        # Normalize the input question.
        question = question.strip()

        # Return an empty list for empty input.
        if not question:
            return []

        # Ask the model to generate multiple search-friendly query variants.
        response = self.client.responses.create(
            model=self.chat_model,
            input=self._build_prompt(question),
        )

        # Extract the plain text output produced by the model.
        rewrite_text = response.output_text.strip()

        # Parse, deduplicate, and limit the generated queries.
        return self._parse_queries(rewrite_text, question, self.num_rewrites)