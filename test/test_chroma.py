from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

embedding = OpenAIEmbeddings(model="text-embedding-3-large")

vectorstore = Chroma(
    collection_name="wwii_chunks",
    persist_directory="data/chroma",
    embedding_function=embedding,
)

docs = vectorstore.similarity_search(
    "Why did Germany invade the Soviet Union?",
    k=5
)

for d in docs:
    print("\n---")
    print(d.page_content[:200])
    print(d.metadata)