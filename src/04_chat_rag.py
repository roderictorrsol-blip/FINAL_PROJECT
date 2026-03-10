r"""
04_chat_rag.py

CLI client for the reusable RAG pipeline.

Run:
  .\.venv\Scripts\python src\04_chat_rag.py
"""

from __future__ import annotations

from pipeline.rag_pipeline import RAGPipeline


DEBUG_CONTEXT = False


def main() -> None:
    pipeline = RAGPipeline()

    print("RAG Chat (CLI) — type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()

        if not question:
            continue

        if question.lower() in {"exit", "quit"}:
            break

        try:
            result = pipeline.run(question)
        except Exception as e:
            print("\nError:", e)
            continue

        print("\n[Debug] queries used:")
        for query in result["queries"]:
            print("-", query)

        print("\nAssistant:\n" + result["answer"] + "\n")

        print("Citations:")
        for c in result["citations"]:
            print("-", c)

        if DEBUG_CONTEXT:
            print("\n[Context sent to LLM]\n")
            print(result["context"])


if __name__ == "__main__":
    main()