# src/evals/02_build_langsmith_dataset.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

DEFAULT_INPUT_PATH = "data/evals/langsmith_eval_candidates.json"
DEFAULT_DATASET_NAME = "wwii-rag-eval"
DEFAULT_DATASET_DESCRIPTION = "Evaluation dataset for WWII transcript-based RAG"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_langsmith_env() -> None:
    if not os.getenv("LANGCHAIN_API_KEY"):
        raise RuntimeError("Missing LANGCHAIN_API_KEY in environment or .env")


def get_or_create_dataset(
    client: Client,
    dataset_name: str,
    description: str,
):
    existing = client.list_datasets(dataset_name=dataset_name)

    for ds in existing:
        if ds.name == dataset_name:
            print(f"[INFO] Using existing dataset: {dataset_name}")
            return ds

    ds = client.create_dataset(
        dataset_name=dataset_name,
        description=description,
    )
    print(f"[OK] Created dataset: {dataset_name}")
    return ds


def build_example(example: Dict[str, Any]) -> Dict[str, Any]:
    question = str(example.get("question", "")).strip()
    reference_answer = str(example.get("reference_answer", "")).strip()

    if not question or not reference_answer:
        raise ValueError("Each example must contain non-empty question and reference_answer")

    metadata = {
        "question_type": example.get("question_type"),
        "source_chunk_id": example.get("source_chunk_id"),
        "source_doc_id": example.get("source_doc_id"),
        "video_id": example.get("video_id"),
        "video_title": example.get("video_title"),
        "source_url_t": example.get("source_url_t"),
        "start_hhmmss": example.get("start_hhmmss"),
    }

    return {
        "inputs": {
            "question": question,
        },
        "outputs": {
            "reference_answer": reference_answer,
        },
        "metadata": metadata,
    }


def main() -> None:
    ensure_langsmith_env()

    base = project_root()
    input_path = base / DEFAULT_INPUT_PATH

    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    raw_examples = load_json(input_path)

    if not isinstance(raw_examples, list) or not raw_examples:
        raise ValueError("langsmith_eval_candidates.json must be a non-empty list")

    dataset_name = os.getenv("LANGSMITH_DATASET_NAME", DEFAULT_DATASET_NAME)
    dataset_description = os.getenv(
        "LANGSMITH_DATASET_DESCRIPTION",
        DEFAULT_DATASET_DESCRIPTION,
    )

    client = Client()
    dataset = get_or_create_dataset(
        client=client,
        dataset_name=dataset_name,
        description=dataset_description,
    )

    prepared_examples: List[Dict[str, Any]] = []
    for row in raw_examples:
        try:
            prepared_examples.append(build_example(row))
        except Exception as e:
            print(f"[WARN] Skipping invalid example: {e}")

    if not prepared_examples:
        raise ValueError("No valid examples found to upload")

    client.create_examples(
        dataset_id=dataset.id,
        inputs=[ex["inputs"] for ex in prepared_examples],
        outputs=[ex["outputs"] for ex in prepared_examples],
        metadata=[ex["metadata"] for ex in prepared_examples],
    )

    print()
    print(f"[OK] Uploaded {len(prepared_examples)} examples to LangSmith")
    print(f"[OK] Dataset name: {dataset.name}")


if __name__ == "__main__":
    main()