"""
In this script we will create the embedding and add those embeddings to the (vector) database.
We use faiss to create the database and to add the embeddings to the database.
We use sentence-transformers to create the embeddings.
"""

import pickle
import os
from tqdm import tqdm

import numpy as np

import faiss
from sentence_transformers import SentenceTransformer, util

from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever
from haystack.schema import Document, FilterType


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


def create_documents_list(data, embeddings):
    """
    Function to create the list of documents (for the document store)

    params:
        data: list of str (list of article)
        embeddings: np.array (array of embeddings) corresponding to the data

    return:
        documents: list of Document (haystack schema)

    """

    # we create the document
    documents = []
    for idx, article in enumerate(data):
        document = Document(content=article, embedding=embeddings[idx, :], id=idx)
        documents.append(document)

    return documents


def create_faiss_document_store(documents, path_index, path_config):
    """
    Create and save faiss document store
    """
    document_store = FAISSDocumentStore(duplicate_documents="overwrite", return_embedding=True)
    document_store.write_documents(documents, duplicate_documents="overwrite")
    document_store.save(index_path=path_index, config_path=path_config)

    return document_store


if __name__ == "__main__":
    # Load the model
    print("Loading the model")
    model = get_model()

    # # Read the data
    # print("Reading the data")
    # data = read_data("../data_preprocess/")

    # # for testing
    # # data = data[:100]

    # # # Create embeddings
    # print("Creating the embeddings")
    # embeddings = compute_embedding_full_text(data, model)

    # embeddings = np.array(embeddings)

    # # # save raw embeddings somewhere (pickle file)
    # print("Saving the embeddings")
    # with open("../embeddings.pickle", "wb") as handle:
    #     pickle.dump(embeddings, handle)

    # load pickle file
    with open("../documents.pickle", "rb") as handle:
        documents = pickle.load(handle)

    # Create the faiss database
    print("Creating the faiss database")
    index = create_faiss_document_store(
        documents, "../faiss_index.index", "../faiss_config.json"
    )

    print(index.get_documents_by_id(["0"]))
