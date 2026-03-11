# src/agents/retriever_chroma.py
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


class RetrieverChroma:
    def __init__(
        self,
        collection_name: str = "wwii_chunks",
        persist_directory: str = "data/chroma",
        embedding_model: str = "text-embedding-3-large",
    ):
        self.embedding = OpenAIEmbeddings(model=embedding_model)
        self.vectorstore = Chroma(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_function=self.embedding,
        )

    def retrieve(self, query: str, k: int = 8):
        return self.vectorstore.similarity_search(query, k=k)