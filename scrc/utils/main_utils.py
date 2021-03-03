import json
import re

import pandas as pd

from scrc.utils.log_utils import get_logger

logger = get_logger()

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# the keys used in the court dataframes
court_keys = [
    "spider",
    "canton",
    "court",
    "chamber",
    "file_name",
    "file_number",
    "file_number_additional",
    "url",
    "date",
    "language",
    "html_raw",
    "html_clean",
    "pdf_raw",
    "pdf_clean",
    "text"
]


def save_to_path(content, path, overwrite=False):
    """
    Create the parent directories of they do not exist.
    If file does not exist already, save content to path.
    :param content:     content to be saved
    :param path:        path of file to be saved
    :param overwrite:   if True overwrites the current content at that path
    :return:
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # check if
    if path.exists():
        logger.debug(f"Path {path} exists already")
        if overwrite:
            logger.debug("Overwrite option specified")
        else:
            return

    logger.debug(f"Saving file to {path}")
    # actually do the saving
    if isinstance(content, bytes):
        path.write_bytes(content)
    elif isinstance(content, str):
        path.write_text(content)
    elif isinstance(content, dict):
        path.write_text(json.dumps(content))
    else:
        raise ValueError(f"Invalid data type {type(content)} supplied.")


def get_raw_text(html) -> str:
    """
    Add the entire text: harder for doing sentence splitting later because of header and footer
    :param html:
    :return:
    """

    raw_text = html.get_text()
    return raw_text


def clean_text(text: str) -> str:
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
