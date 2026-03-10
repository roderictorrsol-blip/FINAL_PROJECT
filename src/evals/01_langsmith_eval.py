from __future__ import annotations

from dotenv import load_dotenv
from langsmith import Client
from langsmith.evaluation import evaluate
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT

from src.evals.datasets.ww2_eval_dataset import EVAL_EXAMPLES
from src.pipeline.rag_pipeline import RAGPipeline


DATASET_NAME = "ww2-rag-expanded-v1"

load_dotenv()

client = Client()
pipeline = RAGPipeline()


def ensure_dataset():
    existing = list(client.list_datasets(dataset_name=DATASET_NAME))
    if existing:
        return existing[0]

    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Base evaluation over WWII_RAG",
    )

    examples = [
        {
            "inputs": {"question": ex["question"]},
            "outputs": {"reference_answer": ex["reference_answer"]},
        }
        for ex in EVAL_EXAMPLES
    ]

    client.create_examples(
        dataset_id=dataset.id,
        examples=examples,
    )

    return dataset


def target(inputs: dict) -> dict:
    question = inputs["question"]
    result = pipeline.run(question)

    docs_text = "\n\n".join(
        getattr(doc, "page_content", "")
        for doc in result.get("docs", [])
    )

    return {
        "answer": result.get("answer", ""),
        "context": docs_text,
    }

correctness_evaluator = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,
    feedback_key="correctness",
    model="openai:gpt-4o-mini",
)

def groundedness_prompt(outputs: dict, reference_outputs: dict | None = None, **kwargs) -> str:
    answer = outputs.get("answer", "")
    context = outputs.get("context", "")

    return f"""Evaluate if the answer is backed by the retireved context.

Context:
{context}

Answer:
{answer}

Return a brief evaluation and a grade.
"""

groundedness_evaluator = create_llm_as_judge(
    prompt=groundedness_prompt,
    feedback_key="groundedness",
    model="openai:gpt-4o-mini",
)

if __name__ == "__main__":
    dataset = ensure_dataset()

    results = evaluate(
        target,
        data=dataset.name,
        evaluators=[
            correctness_evaluator,
            groundedness_evaluator,
        ],
        experiment_prefix="ww2-rag-expanded-v1",
    )

    print("Evaluation correctly launched.")
    print(results)