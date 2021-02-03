import glob
import json
import re

import bs4
import pandas as pd
import spacy

# TODO obige sections zusammennehmen in einen paragraph
# TODO unterteilen nach juristischen Kategorien und anderen
from scrc.utils.main_utils import get_raw_text, clean_text

"""
Data:
Swiss Federal Court Decisions in German: 83732

We are interested in the title and the first div with the class 'content' ('<div class="content">')

"""
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

DATA_PATH = '../data/bger/de/html'

filenames = glob.glob(f"{DATA_PATH}/*-2016.html")  # Here we can also use regex
print(len(filenames))
print(filenames)

# As soon as one of the strings in the list (value) is encountered we switch to the corresponding section (key)
section_markers = {
    "title": ['Urteil vom'],
    "judges": ['Besetzung'],
    "parties": ['Parteien', 'Verfahrensbeteiligte'],
    "topic": ['Gegenstand'],
    "situation": ['Sachverhalt'],
    "considerations": ['Das Bundesgericht zieht in Erwägung', 'Erwägungen'],
    "rulings": ['Demnach erkennt das Bundesgericht'],
    "footer": [r'\w*,\s\d?\d\.\s(?:Jan(?:uar)?|Feb(?:ruar)?|Mär(?:z)?|Apr(?:il)?|Mai|Jun(?:i)?|Jul(?:i)?|Aug(?:ust)?|Sep(?:tember)?|Okt(?:ober)?|Nov(?:ember)?|Dez(?:ember)?).*']
}


def handle_bger():
    data = {'language': [], 'court': [], 'court_id': [], 'file_type': [], 'file_name': [],  # meta information
            'title': [], 'id': [], 'date': [], 'raw_text': [], 'paragraphs': [], }  # content

    file_name = '9C_282-2017.html'

    data['court'].append('Bundesgericht')
    data['court_id'].append('bger')
    data['file_name'].append(file_name)
    data['file_type'].append('html')
    data['language'].append('de')

    with open(file_name) as f:
        html = f.read()
    soup = bs4.BeautifulSoup(html, "html.parser")  # parse html
    divs = soup.find_all("div", class_="content")
    assert len(divs) == 2  # we only expect two divs with class text

    data['title'].append(soup.title.text)  # get the title
    data['id'].append(soup.title.text.split(' ')[0])  # the first part of the title is the case id
    data['date'].append(soup.title.text.split(' ')[1])  # the second part of the title is the date
    raw = get_raw_text(divs[0])
    clean = clean_text(raw)
    data['raw_text'].append(clean)
    paragraphs = get_paragraphs(divs[0])
    data['paragraphs'].append(paragraphs)

    df = pd.DataFrame.from_dict(data)
    print(df)
    df.to_csv('bger.csv')
    return df





def get_paragraphs(html):
    """
    Get Paragraphs in the decision
    :param html:
    :return:
    """
    paragraphs = []
    heading = None
    for div in html:
        if isinstance(div, bs4.element.Tag):
            text = str(div.string)
            # This is a hack to also get tags which contain other tags such as links to BGEs
            if text.strip() == 'None':
                text = div.get_text()
            text = clean_text(text)  # clean the text of unecessary characters
            if "." in text and len(text) < 5:  # get numerated titles such as 1. or A.
                heading = text  # set heading for the next paragraph
            else:
                if heading is not None:  # if we have a heading
                    paragraph = heading + " " + text  # add heading to text of the next paragraph
                else:
                    paragraph = text
                heading = None  # reset heading
            if paragraph not in ['', ' ', ]:  # discard empty paragraphs
                paragraphs.append(paragraph)
    return paragraphs


def bring_to_prodigy_format(df):
    """
    Save content to jsonl file for easy prodigy use.
    :param df:
    :return:
    """
    # make sure it is downloaded with python -m spacy download de_core_news_md
    # nlp = spacy.load("de_core_news_md", disable=["tagger", "textcat", "ner"], n_process=-1)  # only keep parser

    prodigy_data = []

    decision = df.loc[0]

    used_sections = []  # sections which were already found
    current_section = "header"
    for paragraph in decision.paragraphs:
        # update the current section
        for section, markers in section_markers.items():
            if section not in used_sections:  # if we did not use this section yet
                for marker in markers:  # check each marker in the list
                    if re.match(marker, paragraph):
                        current_section = section  # change to the next section
                        used_sections.append(section)  # make sure one section is only used once

        # construct the dict
        paragraph_data = {"text": paragraph, "meta": {
            "section": current_section,
            "court": decision.court,
            "title": decision.title,
            "id": decision.id,
            "date": decision.date,
            "file_name": decision.file_name,
            "file_type": decision.file_type,
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
