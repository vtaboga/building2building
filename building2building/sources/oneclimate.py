"""
OneClimate provides EPW files anywhere on earth.
There is no API, the code scrapes the HTML index pages.
"""

import re
import unicodedata
from pathlib import Path
from typing import Literal
from urllib.parse import urljoin, urlparse

import duckdb
import requests
from building2building.env import STORE_PATH
from building2building.store import (
    OUTPUT,
    ChildFile,
    Derivation,
    DownloadFile,
    ExtractZip,
    derivation,
    realize,
)
from pandas import DataFrame

BASE = (
    "https://climate.onebuilding.org/WMO_Region_4_North_and_Central_America/CAN_Canada/"
)

ProvinceCode = Literal[
    "AB",
    "BC",
    "MB",
    "NB",
    "NL",
    "NS",
    "NT",
    "NU",
    "ON",
    "PE",
    "QC",
    "SK",
    "YT",
]


province_to_code: dict[str, ProvinceCode] = {
    "ALBERTA": "AB",
    "BRITISH COLUMBIA": "BC",
    "MANITOBA": "MB",
    "NEW BRUNSWICK": "NB",
    "NEWFOUNDLAND AND LABRADOR": "NL",
    "NOVA SCOTIA": "NS",
    "NORTHWEST TERRITORIES": "NT",
    "NUNAVUT": "NU",
    "ONTARIO": "ON",
    "PRINCE EDWARD ISLAND": "PE",
    "QUEBEC": "QC",
    "SASKATCHEWAN": "SK",
    "YUKON": "YT",
}


# Province/territory code -> folder on OneBuilding
PROVINCE_FOLDER: dict[ProvinceCode, str] = {
    "AB": "AB_Alberta",
    "BC": "BC_British_Columbia",
    "MB": "MB_Manitoba",
    "NB": "NB_New_Brunswick",
    "NL": "NL_Newfoundland_and_Labrador",
    "NS": "NS_Nova_Scotia",
    "NT": "NT_Northwest_Territories",
    "NU": "NU_Nunavut",
    "ON": "ON_Ontario",
    "PE": "PE_Prince_Edward_Island",
    "QC": "QC_Quebec",
    "SK": "SK_Saskatchewan",
    "YT": "YT_Yukon",
}


@derivation("canada_zip_urls.parquet")
def canada_zip_urls():
    out = OUTPUT.get()

    canada_url = "https://climate.onebuilding.org/WMO_Region_4_North_and_Central_America/CAN_Canada"

    resp = requests.get(canada_url)
    html = resp.text
    hrefs = re.findall(r"href=['\"]([^'\"]+?\.zip)['\"]", html, flags=re.IGNORECASE)

    reverse_province = {
        folder: province for province, folder in PROVINCE_FOLDER.items()
    }

    pattern = (
        r"^(?P<folder>[^/]+)/"
        r"(?P<country_province_city>[^\.]+)\."
        r"(?P<intermediate_tokens>.*)"
        r"\.[^/.]+$"
    )

    rows = []
    for ref in hrefs:
        m = re.match(pattern, ref)
        "CAN_AB_Abee.AgDM.712850_TMYx.zip"

        if m is not None:
            folder = m.group("folder")
            province_code = reverse_province[folder]
            city = m.group("country_province_city")[len("CAN_" + province_code + "_") :]
            url = canada_url + "/" + ref
            rows.append((province_code, city, url))

    df = DataFrame(rows, columns=["province", "city", "url"])
    df.to_parquet(str(out))


def search_weathers(province: ProvinceCode | None = None, city: str | None = None):
    parquet_path = realize(STORE_PATH.get(), canada_zip_urls())
    db = duckdb.from_parquet(str(parquet_path))

    if province is not None:
        db = db.filter(
            duckdb.ColumnExpression("province") == duckdb.ConstantExpression(province)
        )

    if city is not None:
        db = db.filter(
            duckdb.ColumnExpression("city") == duckdb.ConstantExpression(city)
        )

    df = db.to_df()

    def trans(url):
        zip_filename = url.split("/")[-1]
        epw_filename = Path(zip_filename).with_suffix(".epw")

        return lambda: ChildFile(
            ExtractZip(DownloadFile(zip_filename, url, None)), epw_filename
        )

    return df.assign(derivation_thunk=df["url"].apply(trans))
