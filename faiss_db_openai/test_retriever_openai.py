from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever
import os
import openai

# read key.key file and set openai api key
with open("../key.key", "r") as f:
    key = f.read()

# set api_key environment variable
os.environ["api_key"] = key
openai.api_key = os.environ["api_key"]

retriever = EmbeddingRetriever(
    document_store=FAISSDocumentStore.load(
        index_path="faiss_index.index",
        config_path="faiss_config.json",
    ),
    embedding_model="text-embedding-ada-002",
    model_format="openai",
    progress_bar=False,
    api_key=os.environ["api_key"],
)

docs = retriever.retrieve("Syndicaliste et entreprises", top_k=10)
print(docs)
