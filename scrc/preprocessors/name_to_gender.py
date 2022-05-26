from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, Set
from itertools import islice
import json
import re
import datetime
import requests
from scrc.preprocessors.abstract_preprocessor import AbstractPreprocessor
from scrc.utils.log_utils import get_logger
from scrc.utils.main_utils import get_config

if TYPE_CHECKING:
    from sqlalchemy.engine.base import Engine


class NameToGender(AbstractPreprocessor):
    def __init__(self, config: dict):
        super().__init__(config)
        self.logger = get_logger(__name__)
        self.gender_db_file = self.data_dir / "name_to_gender.json"
        self.session = requests.Session()

    def read_file(self):
        self.names_database = json.loads(Path(self.gender_db_file).read_text())

    def read_data_to_match(self, engine: Engine):
        query_str_lang = [
            f"SELECT id, parties, '{lang}' as lang FROM {lang} WHERE parties is not null and parties <> 'null'" for lang in self.languages]
        query_str = ' UNION '.join(query_str_lang)
        return self.query(engine, query_str)

    def check_party_and_representation_for_names(self, current_party, names: Set) -> Set:
        if 'party' in current_party:
            for party in current_party['party']:
                if 'gender' not in party and party['type'] == 'natural person' and 'name' in party and not re.fullmatch(r'[A-Z]\.\_', party['name']):
                    names.add(party['name'])

        if 'representation' in current_party:
            for party in current_party['representation']:
                if 'gender' not in party and party['type'] == 'natural person' and 'name' in party:
                    names.add(party['name'])
        return names

    def start(self):
        self.read_file()

        engine = self.get_engine(self.db_scrc)
        self.data = self.read_data_to_match(engine)

        names = set()

        for idx in range(len(self.data)):
            parties = json.loads(self.data['parties'][idx])
            first_party = parties['0']
            second_party = parties['1']

            names = self.check_party_and_representation_for_names(first_party, names)
            names = self.check_party_and_representation_for_names(second_party, names)


        names = self.filter_names(names)
        names = set(filter(
            lambda n: n not in self.names_database['m'] and n not in self.names_database['f'] and n not in self.names_database['u'], names))
        self.get_gender_from_api(names)

        self.apply_gender_to_data(engine)

    def apply_gender_to_data(self, engine: Engine):
        self.read_file()
        for idx in range(len(self.data)):
            if idx % 10000 == 0:
                self.logger.info(f"Applying gender to data {idx} - {idx+9999} of {len(self.data)}")
            parties = json.loads(self.data['parties'][idx])
            first_party = parties['0']
            second_party = parties['1']
            changed = False
            if 'party' in first_party:
                for party_idx, party in enumerate(first_party['party']):
                    if 'gender' not in party and party['type'] == 'natural person' and 'name' in party and not re.fullmatch(r'[A-Z]\.([A-Z]\.)?\_', party['name']):
                        if party['name'] is not None and party['name'].strip().split()[0] in self.names_database['m']:
                            first_party['party'][party_idx]['gender'] = 'm'
                            changed = True
                        if party['name'] is not None and party['name'].strip().split()[0] in self.names_database['f']:
                            first_party['party'][party_idx]['gender'] = 'f'
                            changed = True

            if 'representation' in first_party:
                for party_idx, party in enumerate(first_party['representation']):
                    if 'gender' not in party and party['type'] == 'natural person' and 'name' in party:
                        if party['name'] is not None and party['name'].strip().split()[0] in self.names_database['m']:
                            first_party['representation'][party_idx]['gender'] = 'm'
                            changed = True
                        if party['name'] is not None and party['name'].strip().split()[0] in self.names_database['f']:
                            first_party['representation'][party_idx]['gender'] = 'f'
                            changed = True

            if 'party' in second_party:
                for party_idx, party in enumerate(second_party['party']):
                    if 'gender' not in party and party['type'] == 'natural person' and 'name' in party and not re.fullmatch(r'[A-Z]\.([A-Z]\.)?\_', party['name']):
                        if party['name'] is not None and party['name'].strip().split()[0] in self.names_database['m']:
                            second_party['party'][party_idx]['gender'] = 'm'
                            changed = True
                        if party['name'] is not None and party['name'].strip().split()[0] in self.names_database['f']:
                            second_party['party'][party_idx]['gender'] = 'f'
                            changed = True

            if 'representation' in second_party:
                for party_idx, party in enumerate(second_party['representation']):
                    if 'gender' not in party and party['type'] == 'natural person' and 'name' in party:
                        if party['name'] is not None and party['name'].strip().split()[0] in self.names_database['m']:
                            second_party['representation'][party_idx]['gender'] = 'm'
                            changed = True
                        if party['name'] is not None and party['name'].strip().split()[0] in self.names_database['f']:
                            second_party['representation'][party_idx]['gender'] = 'f'
                            changed = True

            if changed:
                self.data.loc[idx, 'parties'] = json.dumps({'0': first_party, '1': second_party})
        for language in self.languages:
            self.logger.info(f"Saving table {language}")
            dataframe_language_filtered = self.data[self.data['lang'] == language]
            chunk_size = 1000
            list_df = [dataframe_language_filtered[i:i+chunk_size] for i in range(0,dataframe_language_filtered.shape[0],chunk_size)]
            for index in range(len(list_df)):
                self.logger.info(f"Saving table {language} index {index * chunk_size} - {(index+1) * chunk_size -1}")
                self.update(engine, list_df[index], language, ['parties'], self.output_dir)

    def filter_names(self, names: set[str]) -> set:
        names = [name.strip().split()[0] for name in names if name]
        return set([name for name in names if not '_' in name and len(name) > 3])

    def get_gender_from_api(self, names: Set[str]):
        self.logger.info(f"{len(names)} new names to fetch from the api")
        responses = [self.get_chunk(name_chunk)
                     for name_chunk in self.chunked(names, 10)]
        male = [person['name']
                for responses_chunk in responses for person in responses_chunk if 'gender' in person and person['gender'] == 'male']
        female = [person['name']
                  for responses_chunk in responses for person in responses_chunk if 'gender' in person and person['gender'] == 'female']
        unknown_with_locale = [person['name']
                               for responses_chunk in responses for person in responses_chunk if 'gender' in person and person['gender'] is None]+self.names_database['u']
        responses_without_locale = [self.get_chunk(
            name_chunk, locale=False) for name_chunk in self.chunked(unknown_with_locale, 10)]
        male.extend([person['name']
                    for responses_chunk in responses_without_locale for person in responses_chunk if 'gender' in person and person['gender'] == 'male'])
        female.extend([person['name']
                      for responses_chunk in responses_without_locale for person in responses_chunk if 'gender' in person and person['gender'] == 'female'])
        unknown = [person['name']
                   for responses_chunk in responses_without_locale for person in responses_chunk if 'gender' in person and person['gender'] is None]
        male.extend(self.names_database['m'])
        all_male = set(male)
        female.extend(self.names_database['f'])
        all_female = set(female)
        all_unknown = set(unknown)
        Path(self.gender_db_file).write_text(json.dumps({"m": sorted(
            all_male), "f": sorted(all_female), "u": sorted(all_unknown)}, indent=4))

    def chunked(self, iterable, chunk_size):
        """
        Collect data into chunks of up to length n.
        :type iterable: Iterable[T]
        :type n: int
        :rtype: Iterator[list[T]]
            """
        iterator = iter(iterable)
        while True:
            chunk = list(islice(iterator, chunk_size))
            if chunk:
                yield chunk
            else:
                return

    def get_chunk(self, name_chunk: Set[str], locale: bool = True) -> list:
        params = [('name[]', name) for name in name_chunk]
        if locale:
            params.append(('country_id', 'CH'))
        self.logger.debug('Fetching next chunk of names from genderize')
        response = self.session.get(
            'https://api.genderize.io/',
            params=params,
            timeout=30.0)

        if 'application/json' not in response.headers.get('content-type', ''):
            status = "server responded with {http_code}: {reason}".format(
                http_code=response.status_code, reason=response.reason)
            self.logger.warning(
                'response not in JSON format ({status})'.format(status=status),
                response.headers)
            return [{'name': name, 'gender': None} for name in name_chunk]

        decoded = response.json()
        if response.ok:
            # API returns a single object for a single name
            # but a list for multiple names.
            if not isinstance(decoded, list):
                decoded = [decoded]
            return decoded
        else:
            if response.status_code == 429:
                delta = datetime.timedelta(seconds=int(
                    response.headers['X-Rate-Reset']))
                self.logger.warning(
                    f"Request limit reached. Wait {str(delta)} until {(datetime.datetime.now()+delta).strftime('%d/%m/%Y %H:%M:%S')}")
            return [{'name': name, 'gender': None} for name in name_chunk]


if __name__ == '__main__':
    config = get_config()

    name_to_gender = NameToGender(config)
    name_to_gender.start()
