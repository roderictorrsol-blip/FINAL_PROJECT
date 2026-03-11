# src/evals/01_generate_eval_questions.py
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_INPUT_PATH = "data/chunks/all_chunks_stable.json"
DEFAULT_OUTPUT_PATH = "data/evals/langsmith_eval_candidates.json"

# Number of chunks to sample for automatic question generation
DEFAULT_NUM_CHUNKS = 15

# Number of questions to generate per chunk
DEFAULT_QUESTIONS_PER_CHUNK = 2


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_prompt(chunk: Dict[str, Any], questions_per_chunk: int) -> str:
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
    text = text.strip()

    # Best effort cleanup if the model wraps JSON in markdown fences
    if text.startswith("```"):
        text = text.strip("`")
        lines = text.splitlines()
        if lines and lines[0].lower().startswith("json"):
            lines = lines[1:]
        text = "\n".join(lines)

    data = json.loads(text)

    if not isinstance(data, list):
        raise ValueError("Model output is not a JSON list")

    out: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue

        question = str(item.get("question", "")).strip()
        answer = str(item.get("reference_answer", "")).strip()
        qtype = str(item.get("question_type", "")).strip() or "unknown"

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
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY in environment or .env")

    base = project_root()
    input_path = base / DEFAULT_INPUT_PATH
    output_path = base / DEFAULT_OUTPUT_PATH

    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    chunks = load_json(input_path)
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("all_chunks_stable.json must be a non-empty list")

    client = OpenAI()
    model = os.getenv("EVAL_GEN_MODEL", DEFAULT_MODEL)
    num_chunks = int(os.getenv("EVAL_NUM_CHUNKS", str(DEFAULT_NUM_CHUNKS)))
    questions_per_chunk = int(
        os.getenv("EVAL_QUESTIONS_PER_CHUNK", str(DEFAULT_QUESTIONS_PER_CHUNK))
    )

    sampled_chunks = random.sample(chunks, min(num_chunks, len(chunks)))

    examples: List[Dict[str, Any]] = []

    for i, chunk in enumerate(sampled_chunks, start=1):
        prompt = build_prompt(chunk, questions_per_chunk)

        try:
            response = client.responses.create(
                model=model,
                input=prompt,
            )

            generated = parse_json_list(response.output_text)

            for ex in generated:
                ex["source_chunk_id"] = chunk.get("chunk_id")
                ex["source_doc_id"] = chunk.get("doc_id")
                ex["video_id"] = chunk.get("video_id")
                ex["video_title"] = chunk.get("video_title")
                ex["source_url_t"] = chunk.get("source_url_t")
                ex["start_hhmmss"] = chunk.get("start_hhmmss")

            examples.extend(generated)
            print(f"[OK] Chunk {i}/{len(sampled_chunks)} -> {len(generated)} examples")

        except Exception as e:
            print(f"[WARN] Chunk {i}/{len(sampled_chunks)} failed: {e}")

    examples = dedupe_examples(examples)

    save_json(output_path, examples)

    print()
    print(f"[OK] Saved {len(examples)} evaluation examples to:")
    print(output_path)


if __name__ == "__main__":
    main()