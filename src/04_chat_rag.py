r"""
04_chat_rag.py

Command-line interface for the reusable RAG pipeline.

Run:
  .\.venv\Scripts\python src\04_chat_rag.py
"""
# Enable postponed evaluation of type annotations.
from __future__ import annotations

# Import the main RAG pipeline class used to process user questions.
from pipeline.rag_pipeline import RAGPipeline

# Toggle this flag to print the full context sent to the LLM.
DEBUG_CONTEXT = False


def main() -> None:
    """Run the interactive CLI chat loop."""

    # Create a pipeline instance once and reuse it for the full session.
    pipeline = RAGPipeline()

    # Print a small welcome message for the user.
    print("RAG Chat (CLI) — type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()

        # Ignore empty inputs and ask again.
        if not question:
            continue

        # Stop the program if the user wants to quit.    
        if question.lower() in {"exit", "quit"}:
            break

        try:
            # Send the question to the RAG pipeline and collect the result.
            result = pipeline.run(question)
        except Exception as e:
            # Print a readable error message and continue the session.
            print("\nError:", e)
            continue

        # Show the generated search queries used by the pipeline.    
        print("\n[Debug] queries used:")
        for query in result["queries"]:
            print("-", query)

        # Show the final assistant answer.
        print("\nAssistant:\n" + result["answer"] + "\n")

        # Show the citations returned by the pipeline.
        print("Citations:")
        for c in result["citations"]:
            print("-", c)

        # Optionally print the raw context passed to the language model.
        if DEBUG_CONTEXT:
            print("\n[Context sent to LLM]\n")
            print(result["context"])

# Run the CLI only when this file is executed directly.
if __name__ == "__main__":
    main()