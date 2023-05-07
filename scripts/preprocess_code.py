"""
In this file, we will preprocess the data for the model.
We preprocess the data using unstructure IO.
"""
import re
import pickle
import os
from pdfminer.high_level import extract_text


def preprocess_code(path_pdf, path_preprocess):
    """
    Function that preprocess the data for the model and then save it in a pickle file.
    """

    text = extract_text(path_pdf)

    pattern = r"\n\n Legif\.\s*\n\n Plan\s*\n\n Jp\.C\.Cass\.\s*\n\n Jp\.Appel\s*\n\n Jp\.Admin\.\s*\n\n Juricaf\s*\n\n"
    regex = re.compile(pattern)

    replacement = " ARTICLES_STAMP "
    new_text = regex.sub(replacement, text)

    split_keyword = replacement

    # Define the regex pattern to match each line
    pattern = r".*"

    # Compile the regex pattern
    regex = re.compile(pattern)

    # Split the text into individual lines using the compiled regex pattern
    lines = regex.findall(new_text)

    lines = [line for line in lines if line.strip() != ""]

    # Group the lines into sections based on the split keyword
    sections = []
    current_section = []
    for line in lines:
        if split_keyword in line:
            sections.append(current_section)
            current_section = [line]
        else:
            current_section.append(line)

    articles = []
    # preprocess of the section
    for subsection in sections:
        articles.append(" ".join(subsection))

    # for all the articles we cut the text after one of the keywork :
    # Livre, Titre, Chapitre, Section, Paragraphe, Sous-section, Article
    cut_keyword = [
        "Livre",
        "Titre",
        "Chapitre",
        "Section",
        "Paragraphe",
        "Sous-section",
        "Article",
    ]

    for idx, article in enumerate(articles):
        current_article = article

        for keyword in cut_keyword:
            if keyword in current_article:
                current_article = current_article.split(keyword)[0]

        articles[idx] = current_article

    # we save all the article in a pickle file
    pickle_filename = path_pdf.split("/")[-1].split(".")[0] + ".pickle"

    with open(path_preprocess + pickle_filename, "wb") as handle:
        pickle.dump(articles, handle)

    # save the articles in a pickle file (but we keep only the short articles)
    # we filter the articles that are too long (>2000 characters)
    articles = [article for article in articles if len(article) < 1500]

    pickle_filename = path_pdf.split("/")[-1].split(".")[0] + "_short.pickle"

    # we save all the article in a pickle file
    with open(path_preprocess + pickle_filename, "wb") as handle:
        pickle.dump(articles, handle)

if __name__ == "__main__":
    PATH_PDF = "../data_pdf"
    PATH_PREPROCESS = "../data_preprocess/"
    # we preprocess all the pdf in the folder data_pdf
    for filename in os.listdir(PATH_PDF):
        if filename.endswith(".pdf"):
            print(filename)
            preprocess_code(PATH_PDF + "/" + filename, PATH_PREPROCESS)
