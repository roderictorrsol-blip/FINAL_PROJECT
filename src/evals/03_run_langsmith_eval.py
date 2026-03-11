"""
run_langsmith_eval.py

Run automated LangSmith evaluation for the WWII RAG pipeline.

Purpose:
- Execute the RAG pipeline against a LangSmith dataset
- Compare multiple retrieval backends
- Score answers automatically using LLM-as-judge evaluators
- Track experiments and metadata in LangSmith

Evaluated metrics:
- correctness
- groundedness
"""

from __future__ import annotations

# Standard library imports.
import json
import os
from typing import Any, Dict, List

# Third-party imports.
from dotenv import load_dotenv
from langsmith.evaluation import evaluate
from openai import OpenAI

# Project imports.
from src.pipeline.rag_pipeline import RAGPipeline

# Load environment variables from the local .env file.
load_dotenv()

# Default LangSmith dataset used for evaluation.
DEFAULT_DATASET_NAME = "wwii-rag-eval"

# Prefix used to name LangSmith experiments.
DEFAULT_EXPERIMENT_PREFIX = "wwii-rag"

# Default judge model used for automated scoring.
DEFAULT_JUDGE_MODEL = "gpt-4o-mini"


def ensure_env() -> None:
    """
    Ensure all required API keys are available before running evaluation.
    """
    if not os.getenv("LANGCHAIN_API_KEY"):
        raise RuntimeError("Missing LANGCHAIN_API_KEY in environment or .env")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY in environment or .env")


def build_app() -> RAGPipeline:
    """
    Build and return a fresh RAG pipeline instance.

    A new pipeline is created for each prediction call to keep the evaluation
    runner simple and isolated from previous state.
    """
    return RAGPipeline()


def predict(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prediction function passed to LangSmith evaluate().

    LangSmith will call this function for each dataset example.
    The function:
    - reads the input question
    - runs the RAG pipeline
    - returns answer, context, citations, and retrieval traces
    """
    # Read and normalize the evaluation question.
    question = str(inputs["question"]).strip()

    # Build the application pipeline and run inference.
    pipeline = build_app()
    result = pipeline.run(question)

    # Read the reranked/final documents used for answering.
    docs = result.get("docs", [])

    # Return structured outputs for downstream evaluators.
    return {
        "answer": result.get("answer", ""),
        "citations": result.get("citations", []),
        "context": result.get("context", ""),
        "retrieved_doc_ids": [
            (doc.metadata or {}).get("doc_id")
            for doc in docs
            if (doc.metadata or {}).get("doc_id")
        ],
        "retrieved_chunk_ids": [
            (doc.metadata or {}).get("chunk_id")
            for doc in docs
            if (doc.metadata or {}).get("chunk_id")
        ],
    }


def _judge_json(prompt: str) -> Dict[str, Any]:
    """
    Run an LLM-as-judge prompt and parse the returned JSON.

    The judge model is used by the evaluators to score:
    - correctness
    - groundedness
    """
    # Create the OpenAI client and select the configured judge model.
    client = OpenAI()
    model = os.getenv("EVAL_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)

    # Ask the judge model to produce a JSON grading result.
    response = client.responses.create(
        model=model,
        input=prompt,
    )

    text = response.output_text.strip()

    # Best-effort cleanup if the model wraps JSON inside Markdown fences.
    if text.startswith("```"):
        text = text.strip("`")
        lines = text.splitlines()
        if lines and lines[0].lower().startswith("json"):
            lines = lines[1:]
        text = "\n".join(lines)

    return json.loads(text)


def correctness_evaluator(
    inputs: Dict[str, Any],
    outputs: Dict[str, Any],
    reference_outputs: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluate semantic correctness of the generated answer relative to
    the reference answer stored in the LangSmith dataset.
    """
    # Read normalized evaluation fields.
    question = str(inputs.get("question", "")).strip()
    answer = str(outputs.get("answer", "")).strip()
    reference_answer = str(reference_outputs.get("reference_answer", "")).strip()

    # Build the judge prompt for correctness scoring.
    prompt = f"""
You are grading the correctness of an answer for a World War II QA task.

Score whether the ANSWER is semantically correct relative to the REFERENCE ANSWER.

Guidelines:
- Focus on factual agreement and meaning, not wording.
- Give 1.0 if the answer is fully correct.
- Give 0.5 if it is partially correct or incomplete but not misleading.
- Give 0.0 if it is incorrect, misleading, or fails to answer the question.

Return valid JSON only with this schema:
{{
  "score": 0.0,
  "reasoning": "..."
}}

QUESTION:
{question}

REFERENCE ANSWER:
{reference_answer}

ANSWER:
{answer}
""".strip()

    # Run the judge model and parse its JSON score.
    result = _judge_json(prompt)

    # Return LangSmith-compatible evaluation output.
    return {
        "key": "correctness",
        "score": float(result["score"]),
        "comment": str(result.get("reasoning", "")).strip(),
    }


def groundedness_evaluator(
    inputs: Dict[str, Any],
    outputs: Dict[str, Any],
    reference_outputs: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluate whether the generated answer is supported by the retrieved context.

    This metric focuses on hallucination risk and support from evidence,
    not on agreement with outside knowledge.
    """
    # Read normalized evaluation fields.
    question = str(inputs.get("question", "")).strip()
    answer = str(outputs.get("answer", "")).strip()
    context = str(outputs.get("context", "")).strip()

    # Build the judge prompt for groundedness scoring.
    prompt = f"""
You are grading whether an answer is grounded in the provided context for a World War II QA task.

Score whether the ANSWER is supported by the CONTEXT.

Guidelines:
- Give 1.0 if the answer is fully supported by the context.
- Give 0.5 if the answer is mostly supported but includes minor unsupported inference.
- Give 0.0 if the answer includes unsupported claims or hallucinations.
- Judge support from the CONTEXT only, not from outside knowledge.

Return valid JSON only with this schema:
{{
  "score": 0.0,
  "reasoning": "..."
}}

QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
{answer}
""".strip()

    # Run the judge model and parse its JSON score.
    result = _judge_json(prompt)

    # Return LangSmith-compatible evaluation output.
    return {
        "key": "groundedness",
        "score": float(result["score"]),
        "comment": str(result.get("reasoning", "")).strip(),
    }


def run_for_backend(backend: str) -> None:
    """
    Run a full LangSmith evaluation experiment for a single retrieval backend.

    Supported backends are configured through the RAG pipeline environment:
    - faiss
    - chroma
    - hybrid
    """
    # Set the retrieval backend for this experiment run.
    os.environ["VECTOR_BACKEND"] = backend

    # Read dataset and experiment naming configuration.
    dataset_name = os.getenv("LANGSMITH_DATASET_NAME", DEFAULT_DATASET_NAME)
    experiment_prefix = os.getenv("LANGSMITH_EXPERIMENT_PREFIX", DEFAULT_EXPERIMENT_PREFIX)
    experiment_name = f"{experiment_prefix}-{backend}"

    print(f"[INFO] Running LangSmith eval for backend='{backend}'")
    print(f"[INFO] Dataset: {dataset_name}")
    print(f"[INFO] Experiment: {experiment_name}")

    # Launch the LangSmith evaluation run.
    evaluate(
        predict,
        data=dataset_name,
        experiment_prefix=experiment_name,
        evaluators=[correctness_evaluator, groundedness_evaluator],
        metadata={
            "vector_backend": backend,
            "embed_model": os.getenv("EMBED_MODEL", "text-embedding-3-large"),
            "vector_top_k": os.getenv("VECTOR_TOP_K", "8"),
            "bm25_top_k": os.getenv("BM25_TOP_K", "5"),
            "final_retrieval_k": os.getenv("FINAL_RETRIEVAL_K", "10"),
            "judge_model": os.getenv("EVAL_JUDGE_MODEL", DEFAULT_JUDGE_MODEL),
        },
    )

    print(f"[OK] Finished backend='{backend}'")


def main() -> None:
    """
    Run LangSmith evaluation for all configured retrieval backends.

    The list of backends is controlled via:
        EVAL_BACKENDS=faiss,chroma,hybrid
    """
    # Ensure required API keys are available.
    ensure_env()

    # Read the list of backends to evaluate.
    backends_env = os.getenv("EVAL_BACKENDS", "faiss,chroma,hybrid")
    backends: List[str] = [b.strip().lower() for b in backends_env.split(",") if b.strip()]

    # Run one experiment per backend.
    for backend in backends:
        run_for_backend(backend)


if __name__ == "__main__":
    main()