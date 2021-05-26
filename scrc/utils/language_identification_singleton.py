import re
from pathlib import Path

import fasttext
import requests

from scrc.utils.log_utils import get_logger


class LanguageIdentificationSingleton:
    base_url = 'https://dl.fbaipublicfiles.com/fasttext/supervised-models'
    faster_model_url = 'lid.176.bin'  # faster and slightly more accurate (file size=126MB)
    compressed_model_url = 'lid.176.ftz'  # compressed version of the model (file size=917kB)
    temp_path = "/tmp"

    _instance = None

    def __new__(cls, model: str = 'compressed'):
        if cls._instance is None:
            cls._instance = super(LanguageIdentificationSingleton, cls).__new__(cls)
            # Put any initialization here.
            cls._instance.init(model)
        return cls._instance

    def init(self, model: str = 'compressed'):
        self.logger = get_logger(__name__)
        model_path = self.download_model(model)
        self.model = fasttext.load_model(str(model_path))

    def download_model(self, model: str = 'compressed') -> Path:
        """ Downloads the model if it does not exist yet and returns the path to the model """
        if model == 'compressed':
            chosen_model = self.compressed_model_url
        elif model == 'faster':
            chosen_model = self.faster_model_url
        else:
            self.logger.warn("Please choose either 'compressed' or 'faster'. Falling back to 'compressed' model now.")
            chosen_model = self.compressed_model_url

        model_path = Path(self.temp_path) / chosen_model
        if not model_path.exists():
            url_chosen_model = f"{self.base_url}/{chosen_model}"
            self.logger.info(f"Downloading {model} model from {url_chosen_model} and saving it to {model_path}")
            r = requests.get(url_chosen_model, allow_redirects=True)
            open(model_path, 'wb').write(r.content)
        else:
            self.logger.info(f"{model} model already exists in {model_path}")
        return model_path

    def predict_lang(self, text: str, k: int = 5):
        """IMPORTANT: expects input to be encoded as UTF-8!"""
        assert isinstance(text, str)
        text = re.sub(r"(\r\n|\r|\n)", ' ', text)  # remove all new lines
        predictions = self.model.predict(text, k)  # returns top k matching languages
        return predictions

    def get_lang(self, text: str) -> str:
        """This method can be used to just get the top scoring language directly without probabilities"""
        return self.predict_lang(text, k=1)[0][0][9:]


if __name__ == '__main__':
    lang_id = LanguageIdentificationSingleton()
    lang = lang_id.predict_lang("Hej")
    print(lang)
