"""
OneClimate provides EPW files anywhere on earth.
There is no API, the code scrapes the HTML index pages.
"""

import re
import unicodedata
from urllib.parse import urljoin, urlparse
import requests
from pathlib import Path
import pandas as pd
from building2building.store import DownloadFile, ExtractZip, Derivation, derivation, OUTPUT

BASE = "https://climate.onebuilding.org/WMO_Region_4_North_and_Central_America/CAN_Canada/"

# Province/territory code -> folder on OneBuilding
PROVINCE_FOLDER = {
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

PROVINCE_ABBREV = {
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

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower()

def _pick_zip_url(province: str, city: str) -> str:
    code = PROVINCE_ABBREV.get(province.upper())
    if code not in PROVINCE_FOLDER:
        raise ValueError(f"Unknown province/territory code: {province!r}")
    folder = PROVINCE_FOLDER[code]
    # Most directories have an index.html; if not, the directory URL itself will still work
    index_url = urljoin(BASE, f"{folder}/index.html")

    html = requests.get(index_url, timeout=30).text
    # collect all ZIP links on the page (support both single and double quotes)
    hrefs = re.findall(r"href=['\"]([^'\"]+?\.zip)['\"]", html, flags=re.IGNORECASE)
    if not hrefs:
        # fallback: try the directory URL without explicit index.html
        dir_url = urljoin(BASE, f"{folder}/")
        html = requests.get(dir_url, timeout=30).text
        hrefs = re.findall(r"href=['\"]([^'\"]+?\.zip)['\"]", html, flags=re.IGNORECASE)
        if not hrefs:
            raise RuntimeError(f"No ZIP links found on {index_url}")

    # Simple match: pick the first ZIP whose filename contains the (loosely) normalized city name
    def simple_norm(s: str) -> str:
        # lower, strip accents, remove non-alphanumerics so hyphens/spaces/punctuation don't matter
        s = _norm(s)
        return re.sub(r"[^a-z0-9]+", "", s)

    target = simple_norm(city)
    for href in hrefs:
        fname = urlparse(href).path.split("/")[-1]
        if target and target in simple_norm(fname):
            return urljoin(index_url, href)

    # Fallback: no filename contained the city string; just return the first ZIP found
    # TODO: add a warning here
    return urljoin(index_url, hrefs[0])

def weather_files(province: str, city: str) -> Derivation:

    zip_url = _pick_zip_url(province, city)
    filename = urlparse(zip_url).path.split("/")[-1]
    d = DownloadFile(filename, zip_url, None)
    e = ExtractZip(d)
    return e

@derivation("weathers.parquet")
def WeatherTable(root: Path, province: str, city: str):
    dst = OUTPUT.get()

    epws = sorted([p for p in root.rglob("*.epw")])
    if not epws:
        raise FileNotFoundError(f"No EPW files found in extracted OneClimate archive at {root}")

    df = pd.DataFrame(
        {
            "filename": [p.name for p in epws],
            "province": [province] * len(epws),
            "city": [city] * len(epws),
            "filepath": [str(p) for p in epws],
        }
    )

    df.to_parquet(str(dst))


def weather_table(province: str, city: str) -> Derivation:
    return WeatherTable(weather_files(province, city), province, city)