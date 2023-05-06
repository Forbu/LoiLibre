"""
In this script we will create the embedding and add those embeddings to the (vector) database.
We use faiss to create the database and to add the embeddings to the database.
We use sentence-transformers to create the embeddings.
"""


import pickle
import os
from tqdm import tqdm
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
    for text in tqdm(texts):
        embedding = create_embeddings(text, model)
        embeddings_full_text.append(embedding)

    return embeddings_full_text


def get_model():
    model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")
    return model


def create_faiss_database(embeddings):
    """
    function to initialize the faiss database
    """

    # Define the size of your embedding vectors
    d = 768

    # Create an index with the IndexFlatL2 structure
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)

    return index


def read_data(path):
    """
    Read the data from the pickle files.
    We read all the file that have the .pickle extension in the path folder.
    Also they have to have "short" in the name of the file.
    """
    data = []

    files = os.listdir(path)

    # filter the files that have the .pickle extension
    files = [file for file in files if ".pickle" in file]

    # filter the files that have "short" in the name
    files = [file for file in files if "short" in file]

    # now we loop over the files and we read the data
    for file in files:
        with open(path + file, "rb") as handle:
            data += pickle.load(handle)

    return data


if __name__ == "__main__":
    # Load the model
    print("Loading the model")
    model = get_model()

    # Read the data
    print("Reading the data")
    data = read_data("../data_preprocess/")

    # Create embeddings
    print("Creating the embeddings")
    embeddings = compute_embedding_full_text(data, model)

    # Create the faiss database
    print("Creating the faiss database")
    index = create_faiss_database(embeddings)

    # Save the faiss database
    print("Saving the faiss database")
    faiss.write_index(index, "../faiss_database/faiss_database.index")
