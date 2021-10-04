from typing import Any

"""
This file is used to clean the text of decisions sorted by spiders.
The name of the functions should be equal to the spider! Otherwise, they won't be invocated!
"""


def CH_BGer(soup: Any, namespace: dict) -> Any:
    return soup.find_all("div", class_="content")[0]


def ZH_Verwaltungsgericht(soup: Any, namespace: dict) -> Any:
    # last table contains most of the information => for simplicity disregard the rest
    return soup.findChildren('table', recursive=False)[-1]


def ZH_Sozialversicherungsgericht(soup: Any, namespace: dict) -> Any:
    return soup.find_all("div", class_="cell small-12")[0]  # all the content should be in this div


def CH_BGE(soup: Any, namespace: dict) -> Any:
    for page_number in soup.find_all("div", class_="center pagebreak"):
        page_number.decompose()
    return soup


def VD_FindInfo(soup: Any, namespace: dict) -> Any:
    for table in soup.find_all("table"):
        table.decompose()  # table at the beginning contains logo and name of court
    return soup


def AG_Gerichte(soup: Any, namespace: dict) -> Any:
    for header in soup.find_all("div", class_="header"):
        header.decompose()  # remove headers
    return soup
