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


def handle_bger():
    data = {'language': [], 'court': [], 'court_id': [], 'file_type': [], 'file_name': [], 'title': [], 'text': [], }

    data['language'].append('de')
    data['court'].append('Bundesgericht')
    data['court_id'].append('bger')
    data['file_type'].append('html')
    file_name = 'U_99-2006.html'
    with open(file_name) as f:
        html = f.read()

    soup = bs4.BeautifulSoup(html, "html.parser")  # parse html
    data['file_name'].append(file_name)
    data['title'].append(soup.title.text)  # get the title
    texts = soup.find_all("div", class_="content")
    assert len(texts) == 2  # we only expect two divs with class text
    text = texts[0].get_text()  # the first one contains the decision, the second one the navigation
    text = re.sub(r"\u00a0", ' ', text)  # remove NBSP
    text = re.sub(r"\s+", ' ', text)  # remove all new lines
    text = re.sub(r"_+", '_', text)  # remove duplicate underscores (from anonymisations)
    data['text'].append(text)

    df = pd.DataFrame.from_dict(data)
    print(df)
    df.to_csv('bger.csv')
    return df


def bring_to_prodigy_format(df):
    # make sure it is downloaded with python -m spacy download de_core_news_md
    nlp = spacy.load("de_core_news_md", disable=["tagger", "textcat", "ner"], n_process=-1)  # only keep parser

    prodigy_data = []

    decision = df.loc[0]

    sentences = list(nlp(decision.text).sents)

    for sent in sentences:
        source = f"{decision.court}: {decision.title}"
        sentence_data = {"text": sent.text, "meta": {
            "source": source,
            "court": decision.court,
            "title": decision.title,
            "file_type": decision.file_type,
            "file_name": decision.file_name,
            "language": decision.language,
        }}
        prodigy_data.append(sentence_data)

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
