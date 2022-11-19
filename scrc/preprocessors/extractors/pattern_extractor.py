from __future__ import annotations
from typing import Set, TYPE_CHECKING, Union
from pathlib import Path
import pandas as pd
import sys
import re
from scrc.enums.section import Section
from pandas.core.frame import DataFrame

import bs4

from scrc.enums.language import Language
from scrc.preprocessors.extractors.abstract_extractor import AbstractExtractor
from scrc.preprocessors.extractors.spider_specific.section_splitting_functions import prepare_section_markers
from scrc.utils.log_utils import get_logger
from scrc.utils.main_utils import get_config
from scrc.utils.sql_select_utils import coverage_query, get_total_decisions, join_decision_and_language_on_parameter, \
    where_decisionid_in_list, where_string_spider

if TYPE_CHECKING:
    from sqlalchemy.engine.base import Engine


class PatternExtractor(AbstractExtractor):
    """
    Extracts and counts paragraphs/keywords commonly used across a court. The output can be found in ./data/patterns
    To start the pattern of a court, remove the court from pattern_etraction.txt and run the module. 
    
    If you simply want to check the coverage of a court run the following command:
    
    python -m scrc.preprocessors.extractors.pattern_extractor 1

    If you want to extract the patterns run the following command:
    
    python -m scrc.preprocessors.extractors.pattern_extractor
    """

    def __init__(self, config: dict):
        super().__init__(config, function_name="pattern_extracting_functions", col_name='')
        self.logger = get_logger(__name__)
        self.columns = ['keyword', 'totalcount', 'example']
        self.df = pd.DataFrame(columns=self.columns)
        self.language = {}
        self.currentLanguage = ''
        self.total = 0
        self.counter = 0
        self.spider = ''
        self.limit = 0
        self.dict = {}
        self.end = {}
        self.logger_info = {
            'start': 'Started pattern extraction',
            'finished': 'Finished pattern extraction',
            'start_spider': 'Started pattern extraction for spider',
            'finish_spider': 'Finished pattern extraction for spider',
        }
        self.processed_file_path = self.progress_dir / "pattern_extraction.txt"

    def get_required_data(self, series: pd.DataFrame) -> Union[bs4.BeautifulSoup, str, None]:
        """Returns the data required by the processing functions"""
        html_raw = series['html_raw']
        if pd.notna(html_raw) and html_raw not in [None, '']:
            # Parses the html string with bs4 and returns the body content
            return bs4.BeautifulSoup(html_raw, "html.parser").find('body')
        pdf_raw = series['pdf_raw']
        if pd.notna(pdf_raw) and pdf_raw not in [None, '']:
            return pdf_raw
        return None

    # def get_database_selection_string(self, spider: str, lang: str) -> str:
    #     """Returns the `where` clause of the select statement for the entries to be processed by extractor"""
    #     return f"spider='{spider}'"

    def add_columns(self, engine: Engine):
        return None
    
    def init_dict(self):
        self.dict = {Language.DE: {'count': 0, 'dict': {}}, Language.FR: {
            'count': 0, 'dict': {}}, Language.IT: {'count': 0, 'dict': {}}, Language.EN: {'count': 0, 'dict': {}}, Language.UK: {'count': 0, 'dict': {}}}

    def start_progress(self, engine: Engine, spider: str, lang: str):
        self.processed_amount = 0
        self.total_to_process = self.coverage_get_total(engine, spider, lang)
        if sys.argv[1] != "0":
            if (int(sys.argv[1]) < 1000 and self.total_to_process > 1000):
                self.total_to_process = 1000
            elif self.total_to_process > 1000:
                self.total_to_process = round(int(sys.argv[1]), -3)
        self.logger.info(f"Total: {self.total_to_process}")

    def process_one_df_row(self, series: pd.DataFrame) -> pd.DataFrame:
        """Processes one row of a raw df"""
        namespace = dict()
        # Add the data to the namespace object which is passed to the extraction function
        namespace['html_url'] = series.get('html_url')
        namespace['pdf_url'] = series.get('pdf_url')
        namespace['date'] = series.get('date')
        namespace['language'] = Language(series['language'])
        namespace['id'] = series['decision_id']
        if 'court_string' in series:
            namespace['court'] = series.get('court_string')
        data = self.get_required_data(series)
        paragraphs = self.call_processing_function(series["spider"], data, namespace)
        if paragraphs:    
            self.analyze_structure(paragraphs, namespace)

    def save_data_to_database(self, series: pd.DataFrame, engine: Engine):
        """Splits the data into their respective parts and saves them to the table"""
        
    def get_coverage(self, spider):
        with self.get_engine(self.db_scrc).connect() as conn:
            for lang_key in Language:
                language_key = Language.get_id_value(lang_key.value)
                if language_key != -1:
                    total_result = conn.execute(get_total_decisions(spider, True, language_key)).fetchone()
                    if total_result[0] != 0:
                        self.logger.info(f'Your coverage for {lang_key} of {spider} ({total_result[0]}):')
                        for section_key in Section:
                            if section_key.value != 1:
                                coverage_result = conn.execute(coverage_query(spider, section_key.value, language_key)).fetchone()
                                coverage =  round(coverage_result[0] / total_result[0]  * 100, 2)
                                if not coverage_result[0]:
                                    self.logger.info(f'No sections found for: {section_key}')
                                else:
                                    self.logger.info(f'{section_key} is {coverage}%. Amount: {coverage_result[0]}')
    
    def start_spider_loop(self, spider_list: Set, engine: Engine):
        for spider in spider_list:
            if len(sys.argv) > 1:
                self.get_coverage(spider)
            else:
                self.init_dict()
                self.process_one_spider(engine, spider)
            self.mark_as_processed(self.processed_file_path, spider)


    def select_df(self, engine: str, spider: str) -> str:
        """Returns the `where` clause of the select statement for the entries to be processed by extractor"""
        if self.spider != spider:
            self.counter = 0
        self.spider = spider
        only_given_decision_ids_string = f" AND {where_decisionid_in_list(self.decision_ids)}" if self.decision_ids is not None and len(
            self.decision_ids) > 0 else ""
        
        return self.select(engine,
                           f"file {join_decision_and_language_on_parameter('file_id', 'file.file_id')} " \
                               "LEFT JOIN chamber ON chamber.chamber_id = decision.chamber_id " \
                               "LEFT JOIN court ON court.court_id = chamber.court_id ",
                           f"decision_id, iso_code as language, html_raw, pdf_raw, html_url, pdf_url, '{spider}' as spider, court_string",
                           where=f"file.file_id IN {where_string_spider('file_id', spider)} {only_given_decision_ids_string}",
                           chunksize=self.chunksize)

    def process_one_spider(self, engine: Engine, spider: str):
        self.logger.info(self.logger_info["start_spider"] + " " + spider)
        dfs = self.select_df(self.get_engine(self.db_scrc), spider)
        for df in dfs:
            df = df.apply(self.process_one_df_row, axis="columns")
        self.create_dfs()
        self.logger.info(f"{self.logger_info['finish_spider']} {spider}")

    def analyze_structure(self, paragraphs: list, namespace: dict):
        self.counter += 1
        self.count_total_cases(namespace)
        noDuplicates = list(dict.fromkeys(paragraphs))
        for item in noDuplicates:
            if item is not None and (4 < len(item)):
                self.check_existance(item, namespace)
        if self.counter % 500 == 0:
            self.logger.info(f"Pattern extractor: {self.spider}: {self.counter} processed")

    def create_dfs(self):
        for key in self.dict:
            if self.dict[key]['count'] > 0:
                self.end[key] = pd.DataFrame.from_dict(
                    self.dict[key]['dict'], orient='index')
                self.end[key] = self.drop_rows(self.end[key])
                self.assign_section(
                    self.end[key], {"language": key}, self.dict[key]['count'])

    def count_total_cases(self, namespace: dict):
        self.dict[namespace['language']]['count'] += 1

    def check_existance(self, item: str, namespace):
        item = re.sub('^=+', '', item)
        url = namespace['html_url']
        if url == '':
            url = namespace['pdf_url']
        if item in self.dict[namespace['language']]['dict']:
            self.dict[namespace['language']]['dict'][item]['totalcount'] += 1
        else:
            self.dict[namespace['language']]['dict'][item] = {
                "totalcount": 1, "keyword": item, "url": url}

    def merge_rows(self, indices, df: DataFrame):
        for index in indices[1:]:
            df.at[indices[0],
                  'totalcount'] += df.at[index, 'totalcount']
        df.drop(indices[1:], inplace=True)
        df.reset_index(drop=True, inplace=True)

    def find_matches(self, df: DataFrame, currentIdx: int, indexes: list, currentPattern):
        if not (str(currentPattern) in df.columns):
            df[str(currentPattern)] = 0
        for index in indexes:
            amount = df.at[index, 'totalcount']
            df.at[currentIdx, 'totalcount'] += amount
            df.at[currentIdx, str(currentPattern)] += amount
            df.drop(index, inplace=True)
        return df

    def assign_section(self, df: DataFrame, namespace, count):
        all_section_markers = {
            Language.FR: {
                Section.FACTS: [r'[F,f]ait', r'FAIT'],
                Section.CONSIDERATIONS: [r'[C,c]onsidère', r'[C,c]onsidérant', r'droit', r'DROIT'],
                Section.RULINGS: [r'prononce', r'motifs', r'MOTIFS'],
                Section.FOOTER: [
                    r'\w*,\s(le\s?)?((\d?\d)|\d\s?(er|re|e)|premier|première|deuxième|troisième)\s?(?:janv|févr|mars|avr|mai|juin|juill|août|sept|oct|nov|déc).{0,10}\d?\d?\d\d\s?(.{0,5}[A-Z]{3}|(?!.{2})|[\.])',
                    r'Au nom de la Cour', r'L[e,a] [g,G]reffière', r'l[e,a] [G,g]reffière', r'Le [P,p]résident']
            },
            Language.DE: {
                Section.FACTS: [r'[S,s]achverhalt', r'[T,t]atsachen', r'[E,e]insicht in', r'in Sachen'],
                Section.CONSIDERATIONS: [r'[E,e]rwägung', r"erwägt", "ergeben", r'Erwägungen'],
                Section.RULINGS: [r'erk[e,a]nnt', r'beschlossen', r'verfügt', r'beschliesst', r'entscheidet',
                                  r'prononce', r'motifs'],
                Section.FOOTER: [r'Rechtsmittelbelehrung',
                                 r'^[\-\s\w\(]*,( den| vom)?\s\d?\d\.?\s?(?:Jan(?:uar)?|Feb(?:ruar)?|Mär(?:z)?|Apr(?:il)?|Mai|Jun(?:i)?|Jul(?:i)?|Aug(?:ust)?|Sep(?:tember)?|Okt(?:ober)?|Nov(?:ember)?|Dez(?:ember)?)\s\d{4}([\s]*$|.*(:|Im Namen))',
                                 r'Im Namen des']
            },
            Language.IT: {
                Section.FACTS: [r'(F|f)att(i|o)'],
                Section.CONSIDERATIONS: [r'(C|c)onsiderando', r'(D|d)iritto', r'Visto', r'(C|c)onsiderato'],
                Section.RULINGS: [r'(P|p)er questi motivi'],
                Section.FOOTER: [
                    r'\w*,\s(il\s?)?((\d?\d)|\d\s?(°))\s?(?:gen(?:naio)?|feb(?:braio)?|mar(?:zo)?|apr(?:ile)?|mag(?:gio)|giu(?:gno)?|lug(?:lio)?|ago(?:sto)?|set(?:tembre)?|ott(?:obre)?|nov(?:embre)?|dic(?:embre)?)\s?\d?\d?\d\d\s?([A-Za-z\/]{0,7}):?\s*$'
                ]
            },
            Language.EN: {
                Section.FACTS: [],
                Section.CONSIDERATIONS: [],
                Section.RULINGS: [],
                Section.FOOTER: []
            },
            Language.UK: {
                Section.FACTS: [],
                Section.CONSIDERATIONS: [],
                Section.RULINGS: [],
                Section.FOOTER: []
            }
        }
        section_markers = prepare_section_markers(
            all_section_markers, namespace)
        dfs = {}
        for key in Section:
            if key == Section.HEADER:
                key = 'unknown'
            dfs[key] = pd.DataFrame([], columns=df.columns)
        df.reset_index(inplace=True)
        if 'keyword' in df.columns:
            for index, element in enumerate(df['keyword'].tolist()):
                if index % 100 == 0:
                    self.logger.info(self.get_progress_string(
                        index, df['keyword'].size, "Section assignment: "))
                foundAssigntment = False
                for key in section_markers:
                    if re.search(section_markers[key], element):
                        if len(element) < 35:
                            row = df.loc[index]
                            row['coverage'] = row['totalcount'] / count * 100
                            dfs[key] = dfs[key].append(
                                row, ignore_index=True)
                            foundAssigntment = True
                if not foundAssigntment:
                    row = df.loc[index]
                    row['coverage'] = row['totalcount'] / count * 100
                    dfs['unknown'] = dfs['unknown'].append(
                        row, ignore_index=True)
        self.df_to_csv(dfs, namespace['language'])

    def get_progress_string(self, progress: int, total: int, name: str):
        return f"{name}: {progress} of {total} processed"

    def df_to_csv(self, dfs, lang):
        if lang != Language.EN:
            with pd.ExcelWriter(self.get_path(self.spider, lang)) as writer:
                for key in dfs:
                    if key != Section.FULLTEXT:
                        if 'totalcount' in dfs[key].columns:
                            dfs[key].sort_values(
                                by=['totalcount'], ascending=False).to_excel(writer, sheet_name=str(key),
                                                                             index=False)

    def drop_rows(self, df: DataFrame):
        if 'totalcount' in df.columns:
            df.drop(
                df[df.totalcount < 5].index, inplace=True)
            df.reset_index(drop=True, inplace=True)
        return df

    def get_path(self, spider: str, lang):
        filepath = Path(
            f'data/patterns/{spider}/{spider}_{lang}_section.xlsx')
        filepath.parent.mkdir(parents=True, exist_ok=True)
        return filepath


if __name__ == '__main__':
    config = get_config()
    pattern_extractor = PatternExtractor(config)
    pattern_extractor.start()
