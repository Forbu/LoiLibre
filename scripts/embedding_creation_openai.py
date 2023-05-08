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

import openai

# read key.key file and set openai api key
with open("../key.key", "r") as f:
    key = f.read()

# set api_key environment variable
os.environ["api_key"] = key
openai.api_key = os.environ["api_key"]


def create_embeddings(sentences, model="text-embedding-ada-002"):
    """
    Simple function to create embeddings from a text.

    params:
        sentence: str
        model: sentence-transformers model

    return:
        embedding: np.array
    """

    # Sentences are encoded by calling model.encode()
    embedding = openai.Embedding.create(input=sentences, model=model)

    # retrieve all the embeddings in a list format
    nb_embedding = len(embedding["data"])

    list_embedding = []

    for i in range(nb_embedding):
        list_embedding.append(list(embedding["data"][i]["embedding"]))

    return list_embedding


def compute_embedding_full_text(texts, batch_size=1000):
    """
    Compute the full text embedding for a list of texts.
    """
    embeddings_full_text = []

    # we divide the texts in batch of batch_size
    texts_batched = [
        texts[i : i + batch_size] for i in range(0, len(texts), batch_size)
    ]

    for idx, texts in enumerate(tqdm(texts_batched)):
        embedding = create_embeddings(texts)
        embeddings_full_text += embedding

        # save the embeddings in case of crash
        with open(f"embeddings_full_text_tmp_{idx}.pickle", "wb") as handle:
            pickle.dump(embeddings_full_text, handle)

    return embeddings_full_text


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
    document_store = FAISSDocumentStore(
        duplicate_documents="overwrite", return_embedding=True, embedding_dim=1536
    )
    document_store.write_documents(documents, duplicate_documents="overwrite")
    document_store.save(index_path=path_index, config_path=path_config)

    return document_store


if __name__ == "__main__":
    # Read the data
    print("Reading the data")
    data = read_data("../data_preprocess/")

    # filter the data where there is nothing ''
    data = [article for article in data if article != ""]

    # # Create embeddings
    print("Creating the embeddings")
    embeddings = compute_embedding_full_text(data)

    embeddings = np.array(embeddings)

    # # save raw embeddings somewhere (pickle file)
    print("Saving the embeddings")
    with open("../embeddings.pickle", "wb") as handle:
        pickle.dump(embeddings, handle)

    # load pickle file
    with open("../embeddings.pickle", "rb") as handle:
        embeddings = pickle.load(handle)

    # # Create the documents
    print("Creating the documents")
    documents = create_documents_list(data, embeddings)
    
    # # save documents somewhere (pickle file)
    with open("../documents.pickle", "wb") as handle:
        pickle.dump(documents, handle)

    # load pickle file
    with open("../documents.pickle", "rb") as handle:
        documents = pickle.load(handle)

    # Create the faiss database
    print("Creating the faiss database")
    index = create_faiss_document_store(
        documents, "../faiss_index.index", "../faiss_config.json"
    )

    print(index.get_documents_by_id(["0"]))
