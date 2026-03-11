# test/test_bm25.py
from dotenv import load_dotenv
load_dotenv()

from src.agents.retriever_bm25 import RetrieverBM25

retriever = RetrieverBM25(top_k=5)

queries = [
    "Operation Barbarossa",
    "What happened at Stalingrad?",
    "Why did Germany invade Poland in 1939?",
]

for query in queries:
    print(f"\n=== QUERY: {query} ===")
    docs = retriever.retrieve(query)

    for i, doc in enumerate(docs, start=1):
        print(f"\n[{i}]")
        print(doc.page_content[:250].replace("\n", " "))
        print(doc.metadata)