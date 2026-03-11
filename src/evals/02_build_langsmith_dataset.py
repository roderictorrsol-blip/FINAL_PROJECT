# src/evals/02_build_langsmith_dataset.py
"""
Upload generated evaluation examples to a LangSmith dataset.

Purpose:
- Read locally generated evaluation candidates from JSON
- Validate and normalize each example
- Create or reuse a LangSmith dataset
- Upload examples in LangSmith-compatible format

Input:
- data/evals/langsmith_eval_candidates.json

Output:
- A reusable LangSmith dataset for automated RAG evaluation
"""

from __future__ import annotations

# Standard library imports.
import json
import os
from pathlib import Path
from typing import Any, Dict, List

# Third-party imports.
from dotenv import load_dotenv
from langsmith import Client

# Load environment variables from the local .env file.
load_dotenv()

# Local JSON file containing the candidate evaluation examples.
DEFAULT_INPUT_PATH = "data/evals/langsmith_eval_candidates.json"

# Default LangSmith dataset name used if no custom value is provided.
DEFAULT_DATASET_NAME = "wwii-rag-eval"

# Default description attached to the LangSmith dataset.
DEFAULT_DATASET_DESCRIPTION = "Evaluation dataset for WWII transcript-based RAG"


def project_root() -> Path:
    """
    Resolve the project root directory.

    This assumes the file lives under:
        project_root/src/evals/02_build_langsmith_dataset.py
    """
    return Path(__file__).resolve().parents[2]


def load_json(path: Path) -> Any:
    """
    Load and parse a JSON file from disk.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_langsmith_env() -> None:
    """
    Ensure that the LangSmith API key is available in the environment.
    """
    if not os.getenv("LANGCHAIN_API_KEY"):
        raise RuntimeError("Missing LANGCHAIN_API_KEY in environment or .env")


def get_or_create_dataset(
    client: Client,
    dataset_name: str,
    description: str,
):
    """
    Retrieve an existing LangSmith dataset by name or create it if missing.

    This makes the script idempotent and prevents creating duplicate datasets
    when re-running the upload step.
    """
    # Search for datasets matching the requested name.
    existing = client.list_datasets(dataset_name=dataset_name)

    for ds in existing:
        if ds.name == dataset_name:
            print(f"[INFO] Using existing dataset: {dataset_name}")
            return ds

    # Create the dataset only if it does not already exist.
    ds = client.create_dataset(
        dataset_name=dataset_name,
        description=description,
    )
    print(f"[OK] Created dataset: {dataset_name}")
    return ds


def build_example(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a local evaluation example into the LangSmith example schema.

    Expected local fields:
    - question
    - reference_answer

    Additional provenance fields are stored in metadata.
    """
    # Read and normalize the required fields.
    question = str(example.get("question", "")).strip()
    reference_answer = str(example.get("reference_answer", "")).strip()

    # Enforce the minimum valid schema for upload.
    if not question or not reference_answer:
        raise ValueError("Each example must contain non-empty question and reference_answer")

    # Preserve useful provenance and categorization metadata.
    metadata = {
        "question_type": example.get("question_type"),
        "source_chunk_id": example.get("source_chunk_id"),
        "source_doc_id": example.get("source_doc_id"),
        "video_id": example.get("video_id"),
        "video_title": example.get("video_title"),
        "source_url_t": example.get("source_url_t"),
        "start_hhmmss": example.get("start_hhmmss"),
    }

    # LangSmith examples are structured as inputs, outputs, and metadata.
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
    """
    Upload evaluation examples from local JSON into a LangSmith dataset.

    Pipeline:
    1. Validate LangSmith environment configuration
    2. Load local evaluation candidates
    3. Create or reuse the target dataset
    4. Convert examples to LangSmith format
    5. Upload them to LangSmith
    """
    # Ensure the LangSmith API key is available before continuing.
    ensure_langsmith_env()

    # Resolve the local input file path.
    base = project_root()
    input_path = base / DEFAULT_INPUT_PATH

    # Ensure the local candidate dataset exists.
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    # Load and validate the candidate examples.
    raw_examples = load_json(input_path)

    if not isinstance(raw_examples, list) or not raw_examples:
        raise ValueError("langsmith_eval_candidates.json must be a non-empty list")

    # Read dataset configuration from environment variables when available.
    dataset_name = os.getenv("LANGSMITH_DATASET_NAME", DEFAULT_DATASET_NAME)
    dataset_description = os.getenv(
        "LANGSMITH_DATASET_DESCRIPTION",
        DEFAULT_DATASET_DESCRIPTION,
    )

    # Create the LangSmith client and retrieve or create the target dataset.
    client = Client()
    dataset = get_or_create_dataset(
        client=client,
        dataset_name=dataset_name,
        description=dataset_description,
    )

    # Convert raw local examples into LangSmith-compatible examples.
    prepared_examples: List[Dict[str, Any]] = []

    for row in raw_examples:
        try:
            prepared_examples.append(build_example(row))
        except Exception as e:
            # Skip invalid rows without aborting the whole upload.
            print(f"[WARN] Skipping invalid example: {e}")

    # Ensure at least one valid example is available for upload.
    if not prepared_examples:
        raise ValueError("No valid examples found to upload")

    # Upload all examples to the target LangSmith dataset.
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