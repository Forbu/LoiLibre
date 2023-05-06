"""
In this script we will create the embedding and add those embeddings to the (vector) database.
We use faiss to create the database and to add the embeddings to the database.
We use sentence-transformers to create the embeddings.
"""

from sentence_transformers import SentenceTransformer, util
import faiss

def create_embeddings(sentence, model):
    """
    Simple function to create embeddings from a text.

    params:
        sentence: str
        model: sentence-transformers model

    return:
        embedding: np.array
    """

    # Sentences are encoded by calling model.encode()
    embedding = model.encode(sentence)

    return embedding


def compute_embedding_full_text(texts, model):
    """
    Compute the full text embedding for a list of texts.
    """
    embeddings_full_text = []
    for text in texts:
        embedding = create_embeddings(text, model)
        embeddings_full_text.append(embedding)

    return embeddings_full_text


def get_model():
    model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")
    return model


def create_faiss_database():
    """
    function to initialize the faiss database
    """

    # Define the size of your embedding vectors
    d = 512

    # Create an index with the IndexFlatL2 structure
    index = faiss.IndexFlatL2(d)

    return index
