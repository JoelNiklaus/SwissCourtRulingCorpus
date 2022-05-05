import bs4
import pandas as pd
from scrc.enums.language import Language
from scrc.utils.language_identification_singleton import LanguageIdentificationSingleton
from scrc.preprocessors.abstract_preprocessor import AbstractPreprocessor
from scrc.utils.log_utils import get_logger
from scrc.utils.main_utils import get_config


class LanguageIdentifier(AbstractPreprocessor):
    """ This class cannot run in a parallel fashion as it will throw a PicklingError due to the LanguageIdentificationSingleton """

    def __init__(self, config: dict):
        super().__init__(config)
        self.lang_id = LanguageIdentificationSingleton()
        self.logger = get_logger(__name__)

    def start(self):

        all_decision_ids = []
        # Fetch all decisions with language id -1
        sql_query = 'SELECT html_raw, pdf_raw, decision_id, language_id FROM decision LEFT JOIN file on file.file_id = decision.file_id WHERE decision.language_id = -1'
        df_iterator = pd.read_sql(sql_query, self.get_engine(
            self.db_scrc).connect(), chunksize=self.chunksize)
        df_list = list(df_iterator)

        self.logger.info(f'Identifying language for {len(df_list)} decisions')
        # Get language
        for df in df_list:
            df = df.apply(self.get_lang, axis="columns")
            # Save in db
            self.update(self.get_engine(self.db_scrc), df, 'decision',
                        ['language_id'], self.output_dir, None, 'decision_id')
            all_decision_ids.extend(df['decision_id'])

        return all_decision_ids

    def get_lang(self, series: pd.Series):

        html_raw = series['html_raw']
        pdf_raw = series['pdf_raw']
        language = '--'
        if pdf_raw is not None and len(pdf_raw) > 0:
            language = self.lang_id.get_lang(pdf_raw)
        if html_raw is not None and len(html_raw) > 0:
            soup = bs4.BeautifulSoup(html_raw, "html.parser")  # parse html
            assert soup.find()  # make sure it is valid html
            language = self.lang_id.get_lang(soup.get_text())

        series['language_id'] = Language.get_id_value(language)
        return series


if __name__ == '__main__':
    config = get_config()

    lang_ident = LanguageIdentifier(config)
    lang_ident.start()
