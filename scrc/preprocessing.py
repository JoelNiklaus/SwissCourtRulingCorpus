import json
import re

import bs4
import pandas as pd
import spacy

"""
Data:
Swiss Federal Court Decisions in German: 83732

We are interested in the title and the first div with the class 'content' ('<div class="content">')

"""
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

DATA_PATH = 'data/bger/de/html'

# As soon as one of the strings in the list (value) is encountered we switch to the corresponding section (key)
section_markers = {
    "judges": ['Besetzung'],
    "parties": ['Parteien'],
    "topic": ['Gegenstand'],
    "situation": ['Sachverhalt'],
    "considerations": ['Das Bundesgericht zieht in Erwägung', 'Erwägungen'],
    "rulings": ['Demnach erkennt das Bundesgericht'],
}


def handle_bger():
    data = {'language': [], 'court': [], 'court_id': [], 'file_type': [], 'file_name': [],  # meta information
            'title': [], 'raw_text': [], 'paragraphs': [], }  # content

    file_name = 'U_99-2006.html'

    data['language'].append('de')
    data['court'].append('Bundesgericht')
    data['court_id'].append('bger')
    data['file_type'].append('html')
    data['file_name'].append(file_name)

    with open(file_name) as f:
        html = f.read()
    soup = bs4.BeautifulSoup(html, "html.parser")  # parse html
    divs = soup.find_all("div", class_="content")
    assert len(divs) == 2  # we only expect two divs with class text

    data['title'].append(soup.title.text)  # get the title
    data['raw_text'].append(get_raw_text(divs[0]))
    paragraphs = get_paragraphs(divs[0])
    data['paragraphs'].append(paragraphs)

    df = pd.DataFrame.from_dict(data)
    print(df)
    df.to_csv('bger.csv')
    return df


def get_raw_text(html):
    """
    Add the entire text: harder for doing sentence splitting later because of header and footer
    :param html:
    :return:
    """
    raw_text = html.get_text()  # the first one contains the decision, the second one the navigation
    raw_text = clean_text(raw_text)
    return raw_text


def get_paragraphs(html):
    """
    Get Paragraphs in the decision
    :param html:
    :return:
    """
    paragraphs = []
    for div in html:
        if isinstance(div, bs4.element.Tag):
            text = str(div.string)
            # This is a hack to also get tags which contain other tags such as links to BGEs
            if text.strip() == 'None':
                text = div.get_text()
            paragraph = clean_text(text)
            if paragraph not in ['', ' ',]:  # Only keep information
                paragraphs.append(paragraph)
    return paragraphs


def clean_text(text):
    """
    Clean text from nasty tokens
    :param text:
    :return:
    """
    text = re.sub(r"\u00a0", ' ', text)  # remove NBSP
    text = re.sub(r"\s+", ' ', text)  # remove all new lines
    text = re.sub(r"_+", '_', text)  # remove duplicate underscores (from anonymisations)
    text = text.strip()  # remove leading and trailing whitespace
    return text


def bring_to_prodigy_format(df):
    # make sure it is downloaded with python -m spacy download de_core_news_md
    # nlp = spacy.load("de_core_news_md", disable=["tagger", "textcat", "ner"], n_process=-1)  # only keep parser

    prodigy_data = []

    decision = df.loc[0]

    used_sections = []  # sections which were already found
    current_section = "header"
    for paragraph in decision.paragraphs:
        # update the current section
        for section, markers in section_markers.items():
            for marker in markers:
                if marker in paragraph and section not in used_sections:
                    current_section = section  # change to the next section
                    used_sections.append(section)  # make sure one section is only used once

        # construct the dict
        paragraph_data = {"text": paragraph, "meta": {
            "section": current_section,
            "court": decision.court,
            "title": decision.title,
            "file_type": decision.file_type,
            "file_name": decision.file_name,
            "language": decision.language,
        }}
        prodigy_data.append(paragraph_data)

    """
    sentences = list(nlp(decision.text).sents)

    for sent in sentences:
        source = f"{decision.court}: {decision.title}"
        sentence_data = {"text": sent.raw_text, "meta": {
            "source": source,
            "court": decision.court,
            "title": decision.title,
            "file_type": decision.file_type,
            "file_name": decision.file_name,
            "language": decision.language,
        }}
        prodigy_data.append(sentence_data)
    """

    save_to_jsonl(prodigy_data, 'bger.jsonl')


def save_to_jsonl(prodigy_data, file_name):
    """
    Save list of python dicts to jsonl format
    :param prodigy_data:        the list of python dicts
    :return:
    """
    with open(file_name, 'w') as outfile:
        for entry in prodigy_data:
            json.dump(entry, outfile)
            outfile.write('\n')


def main():
    df = handle_bger()

    bring_to_prodigy_format(df)
    # df.to_json('bger.jsonl', orient='records', lines=True)  # produce jsonl files for prodigy


if __name__ == '__main__':
    main()
