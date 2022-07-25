import pandas as pd

from scrc.dataset_creation.citation_dataset_creator import CitationDatasetCreator
from scrc.utils.log_utils import get_logger
from scrc.utils.main_utils import get_config
import numpy as np
import itertools
from collections import Counter
from root import ROOT_DIR
from pathlib import Path



from sklearn.feature_extraction.text import TfidfTransformer




"""
Dataset to be created:
- contains supreme court cases  
- cols = feature_col and label
- only cases where feature_col text has good length
- Dataset description:
    - train.jsonl:
        contains only the train split of queries.jsonl (years 2000 - 2014)
    - val.jsonl:
        contains only the validation split of queries.jsonl (years 2015 - 2016)
    - test.jsonl:
        contains only the test split of queries.jsonl (years 2017 - 2021)
Set Labels
    - criticality based on ruling citations
        - count citations of bge in bger cases using methods of Doc2Doc
        - set label critical if number of citations found for a bge
        - only bge get citated -> only part of bge will be critical
        - OPTIONAL: distinct between citations depending on when they got cited
Check distribution of data sets
    - distribution among languages
    - distribution among legal areas
    - distribution among cantons
    - is there bias detectable?
"""


class CitationCriticalityDatasetCreator(CitationDatasetCreator):
    """Creates a dataset containing bger cases and sets for each case a criticality label, based how often the case
    was cited in bger cases"""

    def __init__(self, config: dict):
        super().__init__(config)
        self.logger = get_logger(__name__)
        self.debug = False
        self.split_type = "date-stratified"
        self.dataset_name = "citation_criticality_prediction"
        self.feature_cols = ['facts', 'considerations']
        self.dataset_folder = self.create_dir(self.datasets_subdir, self.dataset_name)
        self.load_rulings()

    def get_dataset(self, feature_col, save_reports):
        """get all required data: all bge and bger cases and label bger cases"""
        df = self.get_df(self.get_engine(self.db_scrc), feature_col, 'citations', save_reports)
        print(f"df has {len(df.index)} different decision_id")
        df = self.set_citation_criticality_label(df, feature_col)
        # rename columns
        df = df.rename(columns={'citation_label': "label"})  # normalize column names
        labels, _ = list(np.unique(np.hstack(df.citation_label), return_index=True))
        return df, labels

    def set_citation_criticality_label(self, df, feature_col):
        """set for each bger ruling a label critical or non-critical depending on how often they
                were cited in other cases """
        self.logger.info(f"Processing labeling of citation_criticality")
        bge_references = self.create_critical_bge_list(df)
        # set labels critical as in bge_criticality_dataset_creator
        file_number_match = df.file_number.astype(str).isin(list(bge_references))
        critical_df = df[file_number_match]
        critical_df['bge_label'] = 'critical'
        non_critical_df = df[~file_number_match]
        non_critical_df['bge_label'] = 'non-critical'
        self.logger.info(f"# critical decisions: {len(critical_df.index)}")
        self.logger.info(f"# non-critical decisions: {len(non_critical_df.index)}")
        return critical_df.append(non_critical_df)

    def create_critical_bge_list(self, df):
        citations_df = self.count_citations_of_bger(df)

        # apply for each row a function which returns True if citation amount is bigger than ???
        critical_df = citations_df[citations_df['count'] > 0]

        # create list of bge file numbers
        critical_bger_list = critical_df['bger_reference'].tolist()
        return critical_bger_list

    def count_citations_of_bger(self, df):
        """ create column for df where amount of citations in other cases is counted """
        # get a dict where for each bge ruling is counted how often it was cited by other rulings
        # dict is like {'BGE 125 II 265' : 2 , 'BGE 129 I 281 : 2}
        type_corpus_frequencies_df = self.process_citation(df)
        print(f"tcf_df has {len(type_corpus_frequencies_df.index)} different bge_file_numbers")

        citation_count_df = self.extract_bge_references()
        print(f"citation_df has {len(citation_count_df.index)} different bge_file_numbers")

        # add new column 'count' to dataframe
        citation_count_df['count'] = 0

        # iterate over count values
        a = type_corpus_frequencies_df['count'].unique()
        new_df = pd.DataFrame(columns=citation_count_df.columns)
        for i in a:
            # filter type_corpus_frequencies_df for count = i
            temp_freq_df = type_corpus_frequencies_df[type_corpus_frequencies_df['count'] == i]
            file_number_match = citation_count_df.bge_file_number.astype(str).isin(temp_freq_df['bge_file_number'].astype("string").tolist())
            df_temp = citation_count_df[file_number_match]
            df_temp['count'] = i
            new_df = pd.concat([new_df, df_temp])
        return new_df

    def extract_bge_references(self):
        # get dict of bge references with corresponding bge file name
        bge_references_file_path: Path = ROOT_DIR / 'data' / 'progress' / "bge_references_found.txt"
        if not bge_references_file_path.exists():
            raise Exception("bge references need to be extracted first. Run bge_reference_extractor.")
        references = {}
        with bge_references_file_path.open("r") as f:
            for line in f:
                (bge_file_number, chamber, text) = line.split()
                bge_file_number = bge_file_number.split('_', 5)[3]
                bge_file_number = bge_file_number.replace('-', ' ')
                references[bge_file_number] = f"{chamber} {text}"
        # create data frame of dict
        data = {
            "bge_file_number": list(references.keys()),
            "bger_reference": list(references.values())
        }
        df = pd.DataFrame.from_dict(data)
        assert len(df.index) == len(df.bge_file_number.unique())
        return df

    def process_citation(self, df):
        """find for each bge all citations in other bger"""
        self.logger.info(f"Processing the ruling citations.")
        # need to handle language seperately to add possible missing ruling string in right language
        de_df = df[df['lang'] == 'de']
        fr_df = df[df['lang'] == 'fr']
        it_df = df[df['lang'] == 'it']
        de_df['cit_type'] = df.citations.apply(self.get_citations, lang='de')
        fr_df['cit_type'] = df.citations.apply(self.get_citations, lang='fr')
        it_df['cit_type'] = df.citations.apply(self.get_citations, lang='it')
        df = pd.concat([de_df, fr_df, it_df])

        # we cannot use the ones which have no citations
        # because we drop everything we lose some data, but it is easier, because we know that all the entries rulings citations
        # reset the index so that we don't get out of bounds errors
        df = df.dropna(subset=['cit_type']).reset_index(drop=True)

        self.logger.info(f"Building the term-frequency matrix.")
        df[f"counter"] = df['cit_type'].apply(lambda x: dict(Counter(x)))

        # counts how often a ruling is cited
        # assert no entry with 0 exists
        type_corpus_frequencies = dict(Counter(itertools.chain(*df['cit_type'].tolist())))
        data = {
            "bge_file_number": list(type_corpus_frequencies.keys()),
            "count": list(type_corpus_frequencies.values())
        }
        return pd.DataFrame.from_dict(data)

    def test(self):
        series = pd.Series([{'name': 'ruling', 'text': 'BGE 125 II 265',
                             'url': 'https://www.bger.ch/ext/eurospider/live/de/php/aza/http/index.php?lang=de&type=highlight_simple_query&page=8&from_date=09.06.2004&to_date=28.06.2004&sort=relevance&insertion_date=&top_subcollection_aza=all&query_words=&rank=0&azaclir=aza&highlight_docid=atf%3A%2F%2F125-II-265%3Ade&number_of_ranks=0#page265'},
                            {'name': 'ruling', 'text': 'BGE 129 I 281',
                             'url': 'https://www.bger.ch/ext/eurospider/live/de/php/aza/http/index.php?lang=de&type=highlight_simple_query&page=8&from_date=09.06.2004&to_date=28.06.2004&sort=relevance&insertion_date=&top_subcollection_aza=all&query_words=&rank=0&azaclir=aza&highlight_docid=atf%3A%2F%2F129-I-281%3Ade&number_of_ranks=0#page281'},
                            {'name': 'ruling', 'text': '111 Ia 276',
                             'url': 'https://www.bger.ch/ext/eurospider/live/de/php/aza/http/index.php?lang=de&type=highlight_simple_query&page=8&from_date=09.06.2004&to_date=28.06.2004&sort=relevance&insertion_date=&top_subcollection_aza=all&query_words=&ran'}])
        reply = pd.Series([self.get_citations(series, 'rulings', 'de'), self.get_citations(series, 'rulings', 'de')])
        print(reply)
        reply_second = reply.apply(lambda x: dict(Counter(x)))
        type_corpus_frequencies = dict(Counter(itertools.chain(*reply_second.tolist())))
        print(type_corpus_frequencies)
        keys = type_corpus_frequencies.keys()
        print(keys)
        # works as expected


if __name__ == '__main__':
    config = get_config()

    citation_criticality_dataset_creator = CitationCriticalityDatasetCreator(config)
    # citation_criticality_dataset_creator.test()
    citation_criticality_dataset_creator.create_dataset(sub_datasets=False, kaggle=False, huggingface=False, save_reports=False)
