from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from langsmith.evaluation import evaluate
from openai import OpenAI

from src.pipeline.rag_pipeline import RAGPipeline

load_dotenv()

DEFAULT_DATASET_NAME = "wwii-rag-eval"
DEFAULT_EXPERIMENT_PREFIX = "wwii-rag"
DEFAULT_JUDGE_MODEL = "gpt-4o-mini"


def ensure_env() -> None:
    if not os.getenv("LANGCHAIN_API_KEY"):
        raise RuntimeError("Missing LANGCHAIN_API_KEY in environment or .env")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY in environment or .env")


def build_app() -> RAGPipeline:
    return RAGPipeline()


def predict(inputs: Dict[str, Any]) -> Dict[str, Any]:
    question = str(inputs["question"]).strip()

    pipeline = build_app()
    result = pipeline.run(question)

    docs = result.get("docs", [])

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
    client = OpenAI()
    model = os.getenv("EVAL_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)

    response = client.responses.create(
        model=model,
        input=prompt,
    )

    text = response.output_text.strip()

    # Best-effort parsing if model wraps JSON in fences
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
    question = str(inputs.get("question", "")).strip()
    answer = str(outputs.get("answer", "")).strip()
    reference_answer = str(reference_outputs.get("reference_answer", "")).strip()

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

    result = _judge_json(prompt)

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
    question = str(inputs.get("question", "")).strip()
    answer = str(outputs.get("answer", "")).strip()
    context = str(outputs.get("context", "")).strip()

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

    result = _judge_json(prompt)

    return {
        "key": "groundedness",
        "score": float(result["score"]),
        "comment": str(result.get("reasoning", "")).strip(),
    }


def run_for_backend(backend: str) -> None:
    os.environ["VECTOR_BACKEND"] = backend

    dataset_name = os.getenv("LANGSMITH_DATASET_NAME", DEFAULT_DATASET_NAME)
    experiment_prefix = os.getenv("LANGSMITH_EXPERIMENT_PREFIX", DEFAULT_EXPERIMENT_PREFIX)
    experiment_name = f"{experiment_prefix}-{backend}"

    print(f"[INFO] Running LangSmith eval for backend='{backend}'")
    print(f"[INFO] Dataset: {dataset_name}")
    print(f"[INFO] Experiment: {experiment_name}")

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
    ensure_env()

    backends_env = os.getenv("EVAL_BACKENDS", "faiss,chroma,hybrid")
    backends: List[str] = [b.strip().lower() for b in backends_env.split(",") if b.strip()]

    for backend in backends:
        run_for_backend(backend)


if __name__ == "__main__":
    main()