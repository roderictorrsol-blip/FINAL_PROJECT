# src/evals/01_generate_eval_questions.py
"""
Generate candidate evaluation examples for LangSmith from the canonical chunk dataset.

Purpose:
- Sample chunks from the final RAG knowledge base
- Use an LLM to generate grounded evaluation questions
- Produce reference answers for automated evaluation
- Save the result as a JSON dataset for later upload to LangSmith

Output:
- data/evals/langsmith_eval_candidates.json
"""

from __future__ import annotations

# Standard library imports.
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List

# Third-party imports.
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from the local .env file.
load_dotenv()

# Default OpenAI model used to generate evaluation examples.
DEFAULT_MODEL = "gpt-4o-mini"

# Canonical chunk dataset used as the source for evaluation generation.
DEFAULT_INPUT_PATH = "data/chunks/all_chunks_stable.json"

# Output file containing generated candidate evaluation examples.
DEFAULT_OUTPUT_PATH = "data/evals/langsmith_eval_candidates.json"

# Number of chunks randomly sampled for automatic question generation.
DEFAULT_NUM_CHUNKS = 15

# Number of evaluation questions to generate per sampled chunk.
DEFAULT_QUESTIONS_PER_CHUNK = 2


def project_root() -> Path:
    """
    Resolve the project root directory.

    This assumes the file lives under:
        project_root/src/evals/01_generate_eval_questions.py
    """
    return Path(__file__).resolve().parents[2]


def load_json(path: Path) -> Any:
    """
    Load and parse a JSON file from disk.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    """
    Save a JSON payload to disk using UTF-8 and pretty formatting.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_prompt(chunk: Dict[str, Any], questions_per_chunk: int) -> str:
    """
    Build the prompt used to generate evaluation examples from a source chunk.

    The model is asked to create grounded question-answer pairs that can be
    used later in LangSmith to evaluate the RAG system.
    """
    text = chunk["text"]
    video_title = chunk.get("video_title", "unknown")
    start_hhmmss = chunk.get("start_hhmmss", "unknown")
    source_url_t = chunk.get("source_url_t", "")

    return f"""
You are creating evaluation data for a RAG system about World War II.

Your task:
Generate {questions_per_chunk} high-quality evaluation examples from the source passage below.

Each example must include:
- question
- reference_answer
- question_type

Rules:
- Questions must be answerable ONLY from the passage.
- Do not invent facts not supported by the passage.
- Write clear questions in Spanish.
- Write concise but complete answers in Spanish.
- Prefer historically useful questions:
  - factual
  - causal
  - temporal
  - definitional
  - comparison (only if supported by the passage)
- Avoid trivial wording copied directly from the passage.
- Avoid yes/no questions unless unavoidable.
- Do not ask multiple questions in one.
- Output valid JSON only.
- Return a JSON list.

Source metadata:
- video_title: {video_title}
- start_hhmmss: {start_hhmmss}
- source_url_t: {source_url_t}

Passage:
\"\"\"
{text}
\"\"\"

Expected JSON format:
[
  {{
    "question": "...",
    "reference_answer": "...",
    "question_type": "factual"
  }}
]
""".strip()


def parse_json_list(text: str) -> List[Dict[str, Any]]:
    """
    Parse the model output into a validated list of evaluation examples.

    This function performs a best-effort cleanup in case the model wraps
    the JSON output inside Markdown code fences.
    """
    text = text.strip()

    # Best effort cleanup if the model wraps JSON in Markdown fences.
    if text.startswith("```"):
        text = text.strip("`")
        lines = text.splitlines()
        if lines and lines[0].lower().startswith("json"):
            lines = lines[1:]
        text = "\n".join(lines)

    # Parse the JSON payload returned by the model.
    data = json.loads(text)

    if not isinstance(data, list):
        raise ValueError("Model output is not a JSON list")

    out: List[Dict[str, Any]] = []

    for item in data:
        # Keep only dictionary-shaped items.
        if not isinstance(item, dict):
            continue

        # Read and normalize the required fields.
        question = str(item.get("question", "")).strip()
        answer = str(item.get("reference_answer", "")).strip()
        qtype = str(item.get("question_type", "")).strip() or "unknown"

        # Keep only examples with both question and answer.
        if question and answer:
            out.append(
                {
                    "question": question,
                    "reference_answer": answer,
                    "question_type": qtype,
                }
            )

    return out


def dedupe_examples(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate evaluation examples using the normalized question text.
    """
    seen = set()
    unique: List[Dict[str, Any]] = []

    for ex in examples:
        key = ex["question"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(ex)

    return unique


def main() -> None:
    """
    Generate candidate LangSmith evaluation examples from sampled chunks.

    Pipeline:
    1. Load the canonical chunk dataset
    2. Randomly sample chunks
    3. Ask the model to generate question-answer examples
    4. Attach provenance metadata
    5. Deduplicate examples
    6. Save the generated dataset
    """
    # Ensure the OpenAI API key exists before running generation.
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY in environment or .env")

    # Resolve input and output paths.
    base = project_root()
    input_path = base / DEFAULT_INPUT_PATH
    output_path = base / DEFAULT_OUTPUT_PATH

    # Ensure the canonical chunk dataset exists.
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    # Load and validate the chunk dataset.
    chunks = load_json(input_path)
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("all_chunks_stable.json must be a non-empty list")

    # Create the OpenAI client used for example generation.
    client = OpenAI()

    # Read runtime configuration from environment variables when available.
    model = os.getenv("EVAL_GEN_MODEL", DEFAULT_MODEL)
    num_chunks = int(os.getenv("EVAL_NUM_CHUNKS", str(DEFAULT_NUM_CHUNKS)))
    questions_per_chunk = int(
        os.getenv("EVAL_QUESTIONS_PER_CHUNK", str(DEFAULT_QUESTIONS_PER_CHUNK))
    )

    # Randomly sample a subset of chunks to generate evaluation data from.
    sampled_chunks = random.sample(chunks, min(num_chunks, len(chunks)))

    # Accumulate generated examples across all sampled chunks.
    examples: List[Dict[str, Any]] = []

    for i, chunk in enumerate(sampled_chunks, start=1):
        # Build the generation prompt for the current chunk.
        prompt = build_prompt(chunk, questions_per_chunk)

        try:
            # Ask the model to generate grounded evaluation examples.
            response = client.responses.create(
                model=model,
                input=prompt,
            )

            # Parse the model output into structured examples.
            generated = parse_json_list(response.output_text)

            # Attach provenance metadata so each example can be traced back
            # to the source chunk and video.
            for ex in generated:
                ex["source_chunk_id"] = chunk.get("chunk_id")
                ex["source_doc_id"] = chunk.get("doc_id")
                ex["video_id"] = chunk.get("video_id")
                ex["video_title"] = chunk.get("video_title")
                ex["source_url_t"] = chunk.get("source_url_t")
                ex["start_hhmmss"] = chunk.get("start_hhmmss")

            # Add the generated examples to the global list.
            examples.extend(generated)
            print(f"[OK] Chunk {i}/{len(sampled_chunks)} -> {len(generated)} examples")

        except Exception as e:
            # Continue generation even if one sampled chunk fails.
            print(f"[WARN] Chunk {i}/{len(sampled_chunks)} failed: {e}")

    # Remove duplicate questions across all generated examples.
    examples = dedupe_examples(examples)

    # Save the final candidate evaluation dataset.
    save_json(output_path, examples)

    print()
    print(f"[OK] Saved {len(examples)} evaluation examples to:")
    print(output_path)


if __name__ == "__main__":
    main()