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
from azure.storage.fileshare import ShareServiceClient

try:
    from dotenv import load_dotenv

    load_dotenv()
except:
    pass

list_codes = [
    "code civil",
    "code de commerce",
    

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
langue : français
requête : {query}
question autonome :"""


system_template = {
    "role": "system",
    "content": init_prompt,
}

openai.api_type = "azure"
openai.api_key = os.environ["api_key"]
openai.api_base = os.environ["ressource_endpoint"]
openai.api_version = "2022-12-01"

retriever = EmbeddingRetriever(
    document_store=FAISSDocumentStore.load(
        index_path="./loilibre.faiss",
        config_path="./loilibre.json",
    ),
    embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
    model_format="sentence_transformers",
    progress_bar=False,
)

credential = {
    "account_key": os.environ["account_key"],
    "account_name": os.environ["account_name"],
}

account_url = os.environ["account_url"]
file_share_name = "loilibregpt"

user_id = create_user_id(10)


def filter_sources(df, k_summary=3, k_total=10, source="code civil"):
    assert source in ["ipcc", "ipbes", "all"]

    # Filter by source
    if source == "Code civil":
        df = df.loc[df["source"] == "codecivil"]
    elif source == "ipbes":
        df = df.loc[df["source"] == "IPBES"]
    else:
        pass

    # Separate summaries and full reports
    df_summaries = df.loc[df["report_type"].isin(["SPM", "TS"])]
    df_full = df.loc[~df["report_type"].isin(["SPM", "TS"])]

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
    threshold=0.555,
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
    """
    
    """
    meta = source["meta"]
    return f"""
<div class="card">
    <div class="card-content">
        <h2>Doc {i} - {meta['short_name']} - Page {meta['page_number']}</h2>
        <p>{source['content']}</p>
    </div>
    <div class="card-footer">
        <span>{meta['name']}</span>
        <a href="{meta['url']}#page={meta['page_number']}" target="_blank" class="pdf-link">
            <span role="img" aria-label="Open PDF">🔗</span>
        </a>
    </div>
</div>
"""


def chat(
    user_id: str,
    query: str,
    history: list = [system_template],
    report_type: str = "IPCC",
    threshold: float = 0.555,
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

    if report_type not in ["IPCC", "IPBES"]:
        report_type = "all"
    print("Searching in ", report_type, " reports")
    # if report_type == "All available":
    #     retriever = retrieve_all
    # elif report_type == "IPCC only":
    #     retriever = retrieve_giec
    # else:
    #     raise Exception("report_type arg should be in (All available, IPCC only)")

    reformulated_query = openai.Completion.create(
        engine="climateGPT",
        prompt=get_reformulation_prompt(query),
        temperature=0,
        max_tokens=128,
        stop=["\n---\n", "<|im_end|>"],
    )
    reformulated_query = reformulated_query["choices"][0]["text"]
    reformulated_query, language = reformulated_query.split("\n")
    language = language.split(":")[1].strip()

    sources = retrieve_with_summaries(
        reformulated_query,
        retriever,
        k_total=10,
        k_summary=3,
        as_dict=True,
        source=report_type.lower(),
        threshold=threshold,
    )
    response_retriever = {
        "language": language,
        "reformulated_query": reformulated_query,
        "query": query,
        "sources": sources,
    }

    # docs = [d for d in retriever.retrieve(query=reformulated_query, top_k=10) if d.score > threshold]
    messages = history + [{"role": "user", "content": query}]

    if len(sources) > 0:
        docs_string = []
        docs_html = []
        for i, d in enumerate(sources, 1):
            docs_string.append(
                f"📃 Doc {i}: {d['meta']['short_name']} page {d['meta']['page_number']}\n{d['content']}"
            )
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
            engine="climateGPT",
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
        logs = {
            "user_id": user_id[0],
            "prompt": query,
            "retrived": sources,
            "report_type": report_type,
            "prompt_eng": messages[0],
            "answer": messages[-1]["content"],
            "time": timestamp,
        }

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
        complete_response = "**Pas d'élément trouvé dans les textes de loi. Préciser votre réponse**"
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
    gr.Markdown("<h1><center>Climate Q&A 🌍</center></h1>")
    gr.Markdown(
        "<h4><center>Pose tes questions aux textes de loi ici</center></h4>"
    )

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
                "Quelles sont les options légales pour une personne qui souhaite divorcer, notamment en matière de garde d'enfants et de pension alimentaire ?"
                "Quelles sont les démarches à suivre pour créer une entreprise et quels sont les risques et les responsabilités juridiques associés ?"
                "Comment pouvez-vous m'aider à protéger mes droits d'auteur et à faire respecter mes droits de propriété intellectuelle ?"
                "Quels sont mes droits si j'ai été victime de harcèlement au travail ou de discrimination en raison de mon âge, de ma race ou de mon genre ?"
                "Quelles sont les conséquences légales pour une entreprise qui a été poursuivie pour négligence ou faute professionnelle ?"
                "Comment pouvez-vous m'aider à négocier un contrat de location commercial ou résidentiel, et quels sont mes droits et obligations en tant que locataire ou propriétaire ?"
                "Quels sont les défenses possibles pour une personne accusée de crimes sexuels ou de violence domestique ?"
                "Quelles sont les options légales pour une personne qui souhaite contester un testament ou un héritage ?"
                "Comment pouvez-vous m'aider à obtenir une compensation en cas d'accident de voiture ou de blessure personnelle causée par la négligence d'une autre personne ?"
                "Comment pouvez-vous m'aider à obtenir un visa ou un statut de résident permanent aux États-Unis, et quels sont les risques et les avantages associés ?"
                ],
                [ask_examples_hidden],
                examples_per_page=15,
            )

        with gr.Column(scale=1, variant="panel"):
            gr.Markdown("### Sources")
            sources_textbox = gr.Markdown(show_label=False)

    dropdown_sources = gr.inputs.Dropdown(
        ["IPCC", "IPBES", "IPCC and IPBES"],
        default="IPCC",
        label="Select reports",
    )
    ask.submit(
        fn=chat,
        inputs=[user_id_state, ask, state, dropdown_sources],
        outputs=[chatbot, state, sources_textbox],
    )
    ask.submit(reset_textbox, [], [ask])

    ask_examples_hidden.change(
        fn=chat,
        inputs=[user_id_state, ask_examples_hidden, state, dropdown_sources],
        outputs=[chatbot, state, sources_textbox],
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(
                """
<p><b>Climate change and environmental disruptions have become some of the most pressing challenges facing our planet today</b>. As global temperatures rise and ecosystems suffer, it is essential for individuals to understand the gravity of the situation in order to make informed decisions and advocate for appropriate policy changes.</p>
<p>However, comprehending the vast and complex scientific information can be daunting, as the scientific consensus references, such as <b>the Intergovernmental Panel on Climate Change (IPCC) reports, span thousands of pages</b>. To bridge this gap and make climate science more accessible, we introduce <b>ClimateQ&A as a tool to distill expert-level knowledge into easily digestible insights about climate science.</b></p>
<div class="tip-box">
<div class="tip-box-title">
    <span class="light-bulb" role="img" aria-label="Light Bulb">💡</span>
    How does ClimateQ&A work?
</div>
ClimateQ&A harnesses modern OCR techniques to parse and preprocess IPCC reports. By leveraging state-of-the-art question-answering algorithms, <i>ClimateQ&A is able to sift through the extensive collection of climate scientific reports and identify relevant passages in response to user inquiries</i>. Furthermore, the integration of the ChatGPT API allows ClimateQ&A to present complex data in a user-friendly manner, summarizing key points and facilitating communication of climate science to a wider audience.
</div>
<div class="warning-box">
Version 0.2-beta - This tool is under active development
</div>
"""
            )

        with gr.Column(scale=1):
            gr.Markdown("![](https://i.postimg.cc/fLvsvMzM/Untitled-design-5.png)")
            gr.Markdown(
                "*Source : IPCC AR6 - Synthesis Report of the IPCC 6th assessment report (AR6)*"
            )

    gr.Markdown("## How to use ClimateQ&A")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(
                """
    ### 💪 Getting started
    - In the chatbot section, simply type your climate-related question, and ClimateQ&A will provide an answer with references to relevant IPCC reports.
        - ClimateQ&A retrieves specific passages from the IPCC reports to help answer your question accurately.
        - Source information, including page numbers and passages, is displayed on the right side of the screen for easy verification.
        - Feel free to ask follow-up questions within the chatbot for a more in-depth understanding.
    - ClimateQ&A integrates multiple sources (IPCC and IPBES, … ) to cover various aspects of environmental science, such as climate change and biodiversity. See all sources used below.
    """
            )
        with gr.Column(scale=1):
            gr.Markdown(
                """
    ### ⚠️ Limitations
    <div class="warning-box">
    <ul>
        <li>Please note that, like any AI, the model may occasionally generate an inaccurate or imprecise answer. Always refer to the provided sources to verify the validity of the information given. If you find any issues with the response, kindly provide feedback to help improve the system.</li>
        <li>ClimateQ&A is specifically designed for climate-related inquiries. If you ask a non-environmental question, the chatbot will politely remind you that its focus is on climate and environmental issues.</li>
    </div>
    """
            )

    gr.Markdown("## 🙏 Feedback and feature requests")
    gr.Markdown(
        """
    ### Beta test
    - ClimateQ&A welcomes community contributions. To participate, head over to the Community Tab and create a "New Discussion" to ask questions and share your insights.
    - Provide feedback through email, letting us know which insights you found accurate, useful, or not. Your input will help us improve the platform.
    - Only a few sources (see below) are integrated (all IPCC, IPBES), if you are a climate science researcher and net to sift through another report, please let us know.
    
    If you need us to ask another climate science report or ask any question, contact us at <b>theo.alvesdacosta@ekimetrics.com</b>
    """
    )

    gr.Markdown(
        """
        Little test to see if it works
        """
    )

    demo.queue(concurrency_count=16)

demo.launch()
