import gradio as gr
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever
import openai
import pandas as pd
import os
from utils import (
    make_pairs,
    set_openai_api_key,
    create_user_id,
    to_completion,
)
import numpy as np
from datetime import datetime

try:
    from dotenv import load_dotenv

    load_dotenv()
except:
    pass

list_codes = []

theme = gr.themes.Soft(
    primary_hue="sky",
    font=[gr.themes.GoogleFont("Poppins"), "ui-sans-serif", "system-ui", "sans-serif"],
)

init_prompt = (
    "Vous êtes LoiLibreQA, un assistant AI open source pour l'assistance juridique.",
    "Vous recevez une question et des extraits d'article de loi",
    "Fournissez une réponse claire et structurée en vous basant sur le contexte fourni.",
    "Lorsque cela est pertinent, utilisez des points et des listes pour structurer vos réponses.",
)
sources_prompt = (
    "Lorsque cela est pertinent, utilisez les documents suivants dans votre réponse.",
    "Chaque fois que vous utilisez des informations provenant d'un document, référencez-le à la fin de la phrase (ex : [doc 2]).",
    "Vous n'êtes pas obligé d'utiliser tous les documents, seulement s'ils ont du sens dans la conversation.",
    "Si aucune information pertinente pour répondre à la question n'est présente dans les documents, indiquez simplement que vous n'avez pas suffisamment d'informations pour répondre.",
)


def get_reformulation_prompt(query: str) -> str:
    return f"""Reformulez le message utilisateur suivant en une question courte et autonome en français, dans le contexte d'une discussion autour de questions juridiques.
---
requête: La justice doit-elle être la même pour tous ?
question autonome : Pensez-vous que la justice devrait être appliquée de manière égale à tous, indépendamment de leur statut social ou de leur origine ?
langage: French
---
requête: Comment protéger ses droits d'auteur ?
question autonome : Quelles sont les mesures à prendre pour protéger ses droits d'auteur en tant qu'auteur ?
langage: French
---
requête: Peut-on utiliser une photo trouvée sur Internet pour un projet commercial ?
question autonome : Est-il légalement permis d'utiliser une photographie trouvée sur Internet pour un projet commercial sans obtenir l'autorisation du titulaire des droits d'auteur ?
langage: French
---
requête : {query}
question autonome : """


system_template = {
    "role": "system",
    "content": init_prompt,
}

# read key.key file and set openai api key
with open("key.key", "r") as f:
    key = f.read()

# set api_key environment variable
os.environ["api_key"] = key

set_openai_api_key(key)

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


file_share_name = "loilibregpt"

user_id = create_user_id(10)


def filter_sources(df, k_summary=3, k_total=10, source="code civil"):
    # assert source in ["ipcc", "ipbes", "all"]

    # # Filter by source
    # if source == "Code civil":
    #     df = df.loc[df["source"] == "codecivil"]
    # elif source == "ipbes":
    #     df = df.loc[df["source"] == "IPBES"]
    # else:
    #     pass

    # Separate summaries and full reports
    df_summaries = df  # .loc[df["report_type"].isin(["SPM", "TS"])]
    df_full = df  # .loc[~df["report_type"].isin(["SPM", "TS"])]

    # Find passages from summaries dataset
    passages_summaries = df_summaries.head(k_summary)

    # Find passages from full reports dataset
    passages_fullreports = df_full.head(k_total - len(passages_summaries))

    # Concatenate passages
    passages = pd.concat(
        [passages_summaries, passages_fullreports], axis=0, ignore_index=True
    )
    return passages


def retrieve_with_summaries(
    query,
    retriever,
    k_summary=3,
    k_total=10,
    source="ipcc",
    max_k=100,
    threshold=0.49,
    as_dict=True,
):
    """
    compare to retrieve_with_summaries, this function returns a dataframe with the content of the passages
    """
    assert max_k > k_total
    docs = retriever.retrieve(query, top_k=max_k)
    docs = [
        {**x.meta, "score": x.score, "content": x.content}
        for x in docs
        if x.score > threshold
    ]
    if len(docs) == 0:
        return []
    res = pd.DataFrame(docs)
    passages_df = filter_sources(res, k_summary, k_total, source)
    if as_dict:
        contents = passages_df["content"].tolist()
        meta = passages_df.drop(columns=["content"]).to_dict(orient="records")
        passages = []
        for i in range(len(contents)):
            passages.append({"content": contents[i], "meta": meta[i]})
        return passages
    else:
        return passages_df


def make_html_source(source, i):
    """ """
    meta = source["meta"]
    return f"""
<div class="card">
    <div class="card-content">
        <h2>Doc {i} - </h2>
        <p>{source['content']}</p>
    </div>
    <div class="card-footer">
        <span>link to code</span>
    </div>
</div>
"""


def chat(
    user_id: str,
    query: str,
    history: list = [system_template],
    threshold: float = 0.49,
) -> tuple:
    """retrieve relevant documents in the document store then query gpt-turbo
    Args:
        query (str): user message.
        history (list, optional): history of the conversation. Defaults to [system_template].
        report_type (str, optional): should be "All available" or "IPCC only". Defaults to "All available".
        threshold (float, optional): similarity threshold, don't increase more than 0.568. Defaults to 0.56.
    Yields:
        tuple: chat gradio format, chat openai format, sources used.
    """
    reformulated_query = openai.Completion.create(
        model="text-davinci-002",
        prompt=get_reformulation_prompt(query),
        temperature=0,
        max_tokens=128,
        stop=["\n---\n", "<|im_end|>"],
    )

    reformulated_query = reformulated_query["choices"][0]["text"]
    language = "francais"

    sources = retrieve_with_summaries(
        reformulated_query,
        retriever,
        k_total=10,
        k_summary=3,
        as_dict=True,
        threshold=threshold,
    )

    # docs = [d for d in retriever.retrieve(query=reformulated_query, top_k=10) if d.score > threshold]
    messages = history + [{"role": "user", "content": query}]

    if len(sources) > 0:
        docs_string = []
        docs_html = []
        for i, d in enumerate(sources, 1):
            docs_string.append(f"📃 Doc {i}: \n{d['content']}")
            docs_html.append(make_html_source(d, i))
        docs_string = "\n\n".join(
            [f"Query used for retrieval:\n{reformulated_query}"] + docs_string
        )
        docs_html = "\n\n".join(
            [f"Query used for retrieval:\n{reformulated_query}"] + docs_html
        )
        messages.append(
            {
                "role": "system",
                "content": f"{sources_prompt}\n\n{docs_string}\n\nAnswer in {language}:",
            }
        )

        response = openai.Completion.create(
            model="text-davinci-002",
            prompt=to_completion(messages),
            temperature=0,  # deterministic
            stream=True,
            max_tokens=1024,
        )

        complete_response = ""
        messages.pop()

        messages.append({"role": "assistant", "content": complete_response})
        timestamp = str(datetime.now().timestamp())
        file = user_id[0] + timestamp + ".json"

        for chunk in response:
            if (
                chunk_message := chunk["choices"][0].get("text")
            ) and chunk_message != "<|im_end|>":
                complete_response += chunk_message
                messages[-1]["content"] = complete_response
                gradio_format = make_pairs([a["content"] for a in messages[1:]])
                yield gradio_format, messages, docs_html

    else:
        docs_string = "Pas d'élements juridique trouvé dans les codes de loi"
        complete_response = (
            "**Pas d'élément trouvé dans les textes de loi. Préciser votre réponse**"
        )
        messages.append({"role": "assistant", "content": complete_response})
        gradio_format = make_pairs([a["content"] for a in messages[1:]])
        yield gradio_format, messages, docs_string


def save_feedback(feed: str, user_id):
    if len(feed) > 1:
        timestamp = str(datetime.now().timestamp())
        file = user_id[0] + timestamp + ".json"
        logs = {
            "user_id": user_id[0],
            "feedback": feed,
            "time": timestamp,
        }
        return "Feedback submitted, thank you!"


def reset_textbox():
    return gr.update(value="")


with gr.Blocks(title="LoiLibre Q&A", css="style.css", theme=theme) as demo:
    user_id_state = gr.State([user_id])

    # Gradio
    gr.Markdown("<h1><center>LoiLibre Q&A</center></h1>")
    gr.Markdown("<h4><center>Pose tes questions aux textes de loi ici</center></h4>")

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                elem_id="chatbot", label="LoiLibreQ&A chatbot", show_label=False
            )
            state = gr.State([system_template])

            with gr.Row():
                ask = gr.Textbox(
                    show_label=False,
                    placeholder="Pose ta question ici",
                ).style(container=False)
                ask_examples_hidden = gr.Textbox(elem_id="hidden-message")

            examples_questions = gr.Examples(
                [
                    "Quelles sont les options légales pour une personne qui souhaite divorcer, notamment en matière de garde d'enfants et de pension alimentaire ?",
                    "Quelles sont les démarches à suivre pour créer une entreprise et quels sont les risques et les responsabilités juridiques associés ?",
                    "Comment pouvez-vous m'aider à protéger mes droits d'auteur et à faire respecter mes droits de propriété intellectuelle ?",
                    "Quels sont mes droits si j'ai été victime de harcèlement au travail ou de discrimination en raison de mon âge, de ma race ou de mon genre ?",
                    "Quelles sont les conséquences légales pour une entreprise qui a été poursuivie pour négligence ou faute professionnelle ?",
                    "Comment pouvez-vous m'aider à négocier un contrat de location commercial ou résidentiel, et quels sont mes droits et obligations en tant que locataire ou propriétaire ?",
                    "Quels sont les défenses possibles pour une personne accusée de crimes sexuels ou de violence domestique ?",
                    "Quelles sont les options légales pour une personne qui souhaite contester un testament ou un héritage ?",
                    "Comment pouvez-vous m'aider à obtenir une compensation en cas d'accident de voiture ou de blessure personnelle causée par la négligence d'une autre personne ?",
                    "Comment pouvez-vous m'aider à obtenir un visa ou un statut de résident permanent aux États-Unis, et quels sont les risques et les avantages associés ?",
                ],
                [ask_examples_hidden],
            )

        with gr.Column(scale=1, variant="panel"):
            gr.Markdown("### Sources")
            sources_textbox = gr.Markdown(show_label=False)
            
        

    ask.submit(
        fn=chat,
        inputs=[user_id_state, ask, state],
        outputs=[chatbot, state, sources_textbox],
    )
    ask.submit(reset_textbox, [], [ask])

    ask_examples_hidden.change(
        fn=chat,
        inputs=[user_id_state, ask_examples_hidden, state],
        outputs=[chatbot, state, sources_textbox],
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(
                """
                <div class="warning-box">
                Version 0.2-beta - This tool is under active development
                </div>
                """)
            gr.Markdown(
                """
                
                """)

    demo.queue(concurrency_count=16)

demo.launch(server_name="0.0.0.0")
