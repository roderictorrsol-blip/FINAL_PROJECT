# test/test_retriever_agent.py
from dotenv import load_dotenv
load_dotenv()

from src.agents.retriever_agent import RetrieverAgent

agent = RetrieverAgent()

original_question = "Why did Germany invade Poland in 1939?"
queries = [
    "German invasion of Poland causes",
    "What caused the 1939 invasion of Poland?",
    "Poland invasion 1939 reasons",
]

docs = agent.retrieve(original_question, queries)

print(f"Retrieved docs: {len(docs)}")

for i, doc in enumerate(docs[:10], start=1):
    print(f"\n=== DOC {i} ===")
    print(doc.page_content[:300].replace("\n", " "))
    print(doc.metadata)