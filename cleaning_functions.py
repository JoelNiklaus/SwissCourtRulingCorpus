from typing import Any


def CH_BGEr(soup: Any, namespace: dict) -> Any:
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
