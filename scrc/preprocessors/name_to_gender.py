from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Set
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
        """ Loads the local database file of the already fetched names and their gender """
        self.names_database = json.loads(Path(self.gender_db_file).read_text())

    def read_data_to_match(self, engine: Engine):
        """ Get the data, which is the person_id and names of people that are not anonymized and belong to a natural person """
        table = 'person'
        columns = 'person_id, name'
        where = f"gender IS NULL AND is_natural_person AND NOT name LIKE '%._' "
        return self.select(engine, table, columns, where)

    def start(self):
        engine = self.get_engine(self.db_scrc)
        dfs = self.read_data_to_match(engine)
        for data in dfs:
            # The data is chunked, for every chunk execute one for-loop cycle.
            self.read_file()
            self.logger.info(f"Getting the name for the first chunk of {len(data.index)} people")
            self.data = data
            
            # Get the names which should be applied a gender upon
            self.data['name'] = self.data['name'].apply(self.preprocess_names)
            
            # If the name is already in the file then apply the gender to the name
            self.data['gender'] = self.data['name'].apply(self.get_gender_from_file)
            
            # For every name that still has no gender and is not marked in the file as unknow, fetch from the api
            names = set()
            names.update(self.data.loc[self.data['gender'].isna()]['name'])
            names = names.difference(set(self.names_database['u']))
            if len(names) > 0:
                self.get_gender_from_api(names)

            # Try again to apply the genders from the file to the names as the file is updated now
            self.data['gender'] = self.data['name'].apply(self.get_gender_from_file)
            
            # Update the person in the database
            self.logger.info(f"Saving the gender for {len(self.data.loc[~self.data['gender'].isna()])} people")
            self.update(engine, self.data.loc[~self.data['gender'].isna()], 'person', ['gender'], self.output_dir, index_name='person_id')


    def preprocess_names(self, name: str):
        """ Remove dots from name and try to return the name and not an initial or a title """
        name_parts = name.strip().replace('.', '').split()
        if len(name_parts) == 1 or (len(name_parts[0]) > 1 and name_parts[0] != 'dott' and name_parts[0] != 'Dr'):
            return name_parts[0]
        return name_parts[1]
        
    def get_gender_from_file(self, name: str):
        """ Return the gender if the name is in the male or female section of the local database file """
        if name in self.names_database['m']: return 'm'
        if name in self.names_database['f']: return 'f'
        return None

    def get_gender_from_api(self, names: Set[str]):
        """ Fetch the genders from the api """
        self.logger.info(f"{len(names)} new names to fetch from the api")
        
        def get_by_gender_from_response(responses: List[Dict[str, str]], gender: Optional[str]):
            """ Format the response so the list contains only the supplied gender """
            return [person['name'] for responses_chunk in responses for person in responses_chunk if 'gender' in person and person['gender'] == gender]
        
        # Fetch the gender of names from the api using switzerland as the locale
        responses = [self.get_chunk(name_chunk) for name_chunk in self.chunked(names, 10)]
        male = get_by_gender_from_response(responses, 'male')
        female = get_by_gender_from_response(responses, 'female')
        unknown_with_locale = get_by_gender_from_response(responses, None)+self.names_database['u']
        
        # Fetch the gender of names from the api using switzerland as the locale
        responses_without_locale = [self.get_chunk(name_chunk, locale=False) for name_chunk in self.chunked(unknown_with_locale, 10)]
        male.extend(get_by_gender_from_response(responses_without_locale, 'male'))
        female.extend( get_by_gender_from_response(responses_without_locale, 'female'))
        unknown = get_by_gender_from_response(responses_without_locale, None)
        
        # Extend the fetched gender-name pairs with the locally saved ones so it can be written into the file again.
        male.extend(self.names_database['m'])
        all_male = set(male)
        female.extend(self.names_database['f'])
        all_female = set(female)
        unknown.extend(self.names_database['u'])
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
        """ Make a request to the api with a max of ten names in the set (API-Limit) """
        params = [('name[]', name) for name in name_chunk]
        if locale:
            params.append(('country_id', 'CH')) # Try to match the gender for swiss names
        self.logger.debug('Fetching next chunk of names from genderize')
        response = self.session.get(
            'https://api.genderize.io/',
            params=params,
            timeout=30.0)

        if 'application/json' not in response.headers.get('content-type', ''):
            # Something went wrong
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
                # Rate limit of 1000 names a day is reached
                delta = datetime.timedelta(seconds=int(
                    response.headers['X-Rate-Reset']))
                self.logger.warning(
                    f"Request limit reached. Wait {str(delta)} until {(datetime.datetime.now()+delta).strftime('%d/%m/%Y %H:%M:%S')}")
            return [{'name': name, 'gender': None} for name in name_chunk]


if __name__ == '__main__':
    config = get_config()

    name_to_gender = NameToGender(config)
    name_to_gender.start()
