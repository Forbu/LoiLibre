"""
This is the model function for the server
basicly this is a simple function that call 
the embedding database to see what have context
article that are close to the article that the user want to read.

Then we add those information into the context to feed the information to the lanchain model (chatGPT here).
"""
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain.chains import ConversationalRetrievalChain

from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

import qdrant_client

def generate_context(query, qdrant):
    """
    Function that will generate the context for the chatGPT model
    """
    found_docs = qdrant.similarity_search_with_score(query, k=4)

    # add the context to the query
    context = ""

    for doc in found_docs:
        context += doc["payload"]["text"] + "\n"

    return context

def init_model():
    """
    Function that will init the model
    """
    QA_PROMPT = """Vous êtes un assistant juridique IA utile. Utilisez les éléments de contexte suivants pour répondre à la question de fin.
    Si vous ne connaissez pas la réponse, dites simplement que vous ne savez pas. N'essayez PAS d'inventer une réponse.
    Si la question n'est pas liée au contexte, répondez poliment que vous êtes réglé pour répondre uniquement aux questions liées au contexte.
    Utilisez autant de détails que possible lorsque vous répondez

    {context}

    Question: {question}
    Réponse utile en format markdown:"""

    llm = OpenAI(temperature=0.9)
    prompt = PromptTemplate(
        input_variables=["question", "context"],
        template=None,
    )
    
    qa = ConversationalRetrievalChain.from_llm(llm), 
    
    return llm, prompt

def generate_response(context, query, qdrant, llm, prompt):
    """
    Function that will generate the response for the chatGPT model
    """
    
    answer 


def init_vectorstore(path_db, openai_key):
    """
    Function that will init the vectorstore

    params: path_db: path to the database (str)
    params: openai_key: openai key (str)

    return: qdrant: the vectorstore (Qdrant)
    """

    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002", openai_api_key=openai_key
    )

    client = qdrant_client.QdrantClient(path=path_db, prefer_grpc=True)

    qdrant = Qdrant(
        client=client,
        collection_name="my_documents",
        embedding_function=embeddings.embed_query,
    )

    return qdrant
