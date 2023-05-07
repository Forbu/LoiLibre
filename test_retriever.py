

from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever


retriever = EmbeddingRetriever(
    document_store=FAISSDocumentStore.load(
        index_path="faiss_index.index",
        config_path="faiss_config.json",
    ),
    embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
    model_format="sentence_transformers",
    progress_bar=False,
)

docs = retriever.retrieve("Mort sur la route", top_k=10)
print(docs)
