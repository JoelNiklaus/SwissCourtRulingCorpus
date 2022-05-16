from __future__ import annotations
import configparser
from datetime import datetime
import os
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING, Tuple, Union

import bs4
import pandas as pd
import uuid
from sqlalchemy.sql.expression import text

from sqlalchemy.sql.schema import MetaData, Table


from scrc.enums.language import Language
from scrc.enums.section import Section
from scrc.preprocessors.abstract_preprocessor import AbstractPreprocessor
from scrc.preprocessors.extractors.abstract_extractor import AbstractExtractor
from root import ROOT_DIR
from scrc.utils.log_utils import get_logger
from scrc.utils.main_utils import get_config
from scrc.utils.sql_select_utils import delete_stmt_decisions_with_df, join_decision_and_language_on_parameter, where_decisionid_in_list, where_string_spider

if TYPE_CHECKING:
    from sqlalchemy.engine.base import Engine


class SectionSplitter(AbstractExtractor):
    """
    Splits the raw html into sections which can later separately be used. This represents the section splitting task.
    """

    def __init__(self, config: dict):
        super().__init__(config, function_name='section_splitting_functions', col_name='sections')
        self.logger = get_logger(__name__)
        self.logger_info = {
            'start': 'Started section splitting',
            'finished': 'Finished section splitting',
            'start_spider': 'Started splitting sections for spider',
            'finish_spider': 'Finished splitting sections for spider',
            'saving': 'Saving chunk of recognized sections',
            'processing_one': 'Splitting sections from file',
            'no_functions': 'Not splitting into sections.'
        }

        # spacy_tokenizer, bert_tokenizer = tokenizers['de']
        self.tokenizers: dict[Language, Tuple[any, any]] = {
            Language.DE: self.get_tokenizers('de'),
            Language.FR: self.get_tokenizers('fr'),
            Language.IT: self.get_tokenizers('it'),
        }

        self.processed_file_path = self.progress_dir / "spiders_section_split.txt"

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

    def select_df(self, engine: str, spider: str) -> str:
        """Returns the `where` clause of the select statement for the entries to be processed by extractor"""
        only_given_decision_ids_string = f" AND {where_decisionid_in_list(self.decision_ids)}" if self.decision_ids is not None and len(
            self.decision_ids) > 0 else ""
        return self.select(engine, f"file {join_decision_and_language_on_parameter('file_id', 'file.file_id')} LEFT JOIN chamber ON chamber.chamber_id = decision.chamber_id ", f"decision_id, iso_code as language, html_raw, pdf_raw, html_url, pdf_url, '{spider}' as spider, court_id", where=f"file.file_id IN {where_string_spider('file_id', spider)} {only_given_decision_ids_string}", chunksize=self.chunksize)

    def run_tokenizer(self, df: pd.DataFrame):
        # only calculate this if we save the reports because it takes a long time
        # TODO precompute this in previous step after section splitting for each section
        # calculate both the num_tokens for regular words and subwords for the feature_col (not only the entire text)
        spacy_tokenizer, bert_tokenizer = self.tokenizers.get(
            Language(df['language'][0]))

        result = {}

        for idx, row in df.iterrows():
            if not row['sections']:
                return {}
            if len(row['sections']) > 0:
                row['sections'][Section.FULLTEXT] = '\n\n'.join(['\n'.join(
                    row['sections'][section]) for section in row['sections']])
            for k in row['sections'].keys():
                result[k] = {}
                self.logger.debug("Started tokenizing with spacy")
                result[k]['num_tokens_spacy'] = [
                    len(result) for result in spacy_tokenizer.pipe(row['sections'][k], batch_size=100)]
                self.logger.debug("Started tokenizing with bert")
                result[k]['num_tokens_bert'] = [len(input_id) for input_id in bert_tokenizer(
                    df['sections'][k].tolist()).input_ids]

        return result

    def save_data_to_database(self, df: pd.DataFrame, engine: Engine):

        if not AbstractPreprocessor._check_write_privilege(engine):
            path = ''
            AbstractPreprocessor.create_dir(self.output_dir, os.getlogin())
            path = Path.joinpath(self.output_dir, os.getlogin(
            ), datetime.now().isoformat() + '.json')

            with path.open("a") as f:
                df.to_json(f)
            return

        #tokens = self.run_tokenizer(df)

        for idx, row in df.iterrows():
            if idx % 50 == 0:
                self.logger.info(f'Saving decision {idx+1} from chunk')
            if row['sections'] is None or row['sections'].keys is None:
                continue
            with self.get_engine(self.db_scrc).connect() as conn:
                t = Table('section', MetaData(), autoload_with=engine)
                t_paragraph = Table('paragraph', MetaData(),
                                    autoload_with=engine)
                t_num_tokens = Table(
                    'num_tokens', MetaData(), autoload_with=engine)

                # Delete and reinsert as no upsert command is available. This pattern is used multiple times in this method
                stmt = t.delete().returning(text('section_id')).where(
                    delete_stmt_decisions_with_df(df))
                section_ids_result = conn.execute(stmt).all()
                section_ids = [i['section_id'] for i in section_ids_result]
                stmt = t_paragraph.delete().where(delete_stmt_decisions_with_df(df))
                conn.execute(stmt)
                if len(section_ids) > 0:
                    section_ids_string = ','.join(
                        ["'" + str(item)+"'" for item in section_ids])
                    stmt = t_num_tokens.delete().where(
                        text(f"section_id in ({section_ids_string})"))
                    conn.execute(stmt)
                for k in row['sections'].keys():
                    section_type_id = k.value
                    # insert section
                    stmt = t.insert().returning(text("section_id")).values([{"decision_id": str(
                        row['decision_id']), "section_type_id": section_type_id, "section_text": row['sections'][k]}])
                    section_id = conn.execute(stmt).fetchone()['section_id']

                    # Add num tokens
                    # stmt = t_num_tokens.insert().values([{'section_id': str(
                    #    section_id), 'num_tokes_spacy': tokens[k]['num_tokens_spacy'], 'num_tokens_bert': tokens[k]['num_tokens_bert']}])
                    # conn.execute(stmt)

                    # Add a all paragraphs
                    for paragraph in row['sections'][k]:
                        stmt = t_paragraph.insert().values([{'section_id': str(section_id), "decision_id": str(
                            row['decision_id']), 'paragraph_text': paragraph, 'first_level': None, 'second_level': None, 'third_level': None}])
                        conn.execute(stmt)

    def read_column(self, engine: Engine, spider: str, name: str, lang: str) -> pd.DataFrame:
        query = f"SELECT count({name}) FROM {lang} WHERE {self.get_database_selection_string(spider, lang)} AND {name} <> ''"
        return pd.read_sql(query, engine.connect())['count'][0]

    def log_coverage_from_json(self, engine: Engine, spider: str, lang: str, batch_info: dict) -> None:
        """ log_coverage for users without database write privileges, parses results from json logs
       allows to print coverage half way through with the coverage for the parsed chunks still being valid"""

        if self.total_to_process == 0:
            # no entries to process
            return

        path = Path.joinpath(self.output_dir, os.getlogin(),
                             spider, lang, batch_info['uuid'])

        # summary: all collected elements and the count of found sections, initialized to zero
        summary = {'total_collected': 0}
        for section in Section:
            summary[section.value] = 0

        # retrieve all chunks iteratively to limit memory usage
        for chunk in range(batch_info['chunknumber']):
            if not (path/f"{chunk}.json").exists():
                continue
            df_chunk = pd.read_json(str(path / f"{chunk}.json"))
            summary['total_collected'] += df_chunk.shape[0]
            for section in Section:
                # count the amount of found sections if the section is not empty
                if df_chunk.notna()[section.value].any():
                    summary[section.value] += int(
                        (df_chunk[section.value] != '').sum())

        if summary['total_collected'] == 0:
            self.logger.info(
                f"Could not find any stored log files for batch {batch_info['uuid']} in {lang}")
            return

        total_processed = self.total_to_process
        # ceil division to get the number of files (chunks) that should be present if pipeline finished
        total_chunks = -(self.total_to_process // - self.chunksize)
        # if the pipeline was interrupted, notify the user, this allows to print coverage half way through
        # with the coverage for the parsed chunks still being valid
        if total_chunks > batch_info['chunknumber']:
            self.logger.info("The pipeline was interrupted or logged intermittently, "
                             f"the last chunk was {batch_info['uuid']} with number {str(batch_info['chunknumber'])}")
            self.logger.info(f"Coverage for the processed chunks:")
            # update total to give a correct coverage
            total_processed = summary['total_collected']
        else:
            self.logger.info(
                f"{self.logger_info['finish_spider']} in {lang} with the following amount recognized:")
        for section in Section:
            section_amount = summary[section.value]
            self.logger.info(
                f"{section.value.capitalize()}:\t{section_amount} / {total_processed} "
                f"({section_amount / total_processed:.2%}) "
            )

    def log_coverage(self, engine: Engine, spider: str, lang: str):
        """Override method to get custom coverage report"""
        self.logger.info(
            f"{self.logger_info['finish_spider']} in {lang} with the following amount recognized:")
        for section in Section:
            section_amount = self.read_column(
                engine, spider, section.value, lang)
            self.logger.info(
                f"{section.value.capitalize()}:\t{section_amount} / {self.total_to_process} "
                f"({section_amount / self.total_to_process:.2%}) "
            )

    """  def process_one_spider(self, engine: Engine, spider: str):
        self.logger.info(self.logger_info['start_spider'] + ' ' + spider)

        for lang in self.languages:
            where = self.get_database_selection_string(spider, lang)
            self.start_progress(engine, spider, lang)
            # stream dfs from the db
            dfs = self.select(engine, lang, where=where, chunksize=self.chunksize)
            # passing batch_info to store results in json readable for the coverage report if readonly user
            batchinfo = {'uuid': uuid.uuid4().hex[:8], 'chunknumber': 0}
            log_dir = Path.joinpath(self.output_dir, os.getlogin(), spider, lang, batchinfo['uuid'])
            for df in dfs:
                df = df.apply(self.process_one_df_row, axis='columns')
                self.save_data_to_database(df)
                self.update(engine, df, lang, [section.value for section in Section] + ['paragraphs'], self.output_dir)
                self.log_progress(self.chunksize)
                batchinfo['chunknumber'] += 1
                self.log_coverage_from_json(engine, spider, lang, batchinfo)
                
            if self._check_write_privilege(engine):
                self.log_coverage(engine, spider, lang)
            else:
                self.log_coverage_from_json(engine, spider, lang, batchinfo)

        self.logger.info(f"{self.logger_info['finish_spider']} {spider}") 
    """

    """  def process_one_df_row(self, series: pd.DataFrame) -> pd.DataFrame:
        Override method to handle section data and paragraph data individually
        # TODO consider removing the overriding function altogether with new db
        self.logger.debug(f"{self.logger_info['processing_one']} {series['file_name']}")
        namespace = series[['date', 'html_url', 'pdf_url', 'court', 'id']].to_dict()
        namespace['language'] = Language(series['language'])
        data = self.get_required_data(series)
        assert data
        try:
            paragraphs_by_section = self.call_processing_function(series['spider'], data, namespace) or None
            if paragraphs_by_section:
                for section, value in paragraphs_by_section.items():
                    series[section.value] = "\t".join(value)  # TODO save as list in new db
        except TypeError as e:
            self.logger.error(f"While processing decision {series['html_url']} caught exception {e}")
        return series
    """


if __name__ == '__main__':
    config = get_config()
    section_splitter = SectionSplitter(config)
    section_splitter.start()
