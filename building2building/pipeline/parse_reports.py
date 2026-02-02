import re
from io import StringIO
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup


def _parse_float_cell(x: str) -> float | None:
    s = str(x).strip()
    if not s or s.lower() in ("&nbsp;", "unknown"):
        return None
    try:
        return float(s)
    except Exception:
        return None


def get_net_conditioned_area(html_path: Path) -> float:
    """
    Extract 'Net Conditioned Building Area' [m²] from an EnergyPlus
    HTML summary report (eplusout.html / eplusbl.htm / eplustbl.htm).

    Returns:
        area_m2 (float) or None if not found.
    """
    with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f, "html.parser")

    # Find <b>Building Area</b> and get its following <table>
    table_tag = None
    for b in soup.find_all("b"):
        if "Building Area" in b.text:
            table_tag = b.find_next("table")
            break
    if table_tag is None:
        raise Exception("could not find conditionned area")

    # Use StringIO to satisfy pandas future behavior
    df = pd.read_html(StringIO(str(table_tag)))[0]

    # Clean headers robustly (convert to strings first)
    df.columns = [str(col).strip() for col in df.columns]

    # Look for the "Net Conditioned Building Area" row (case-insensitive)
    mask = (
        df.iloc[:, 0]
        .astype(str)
        .str.contains("Net Conditioned Building Area", case=False)
    )
    if not mask.any():
        raise Exception("could not find conditionned area")

    # Extract and return the numeric value (2nd column)
    value = float(df.loc[mask].iloc[0, 1])
    return value


def get_warmup_days(html_path: Path) -> int:
    """
    Extract the number of warm-up days from an EnergyPlus HTML summary file.

    Note: Some EnergyPlus versions / output configurations may not include the
    warmup-days table in `eplustbl.htm`. In that case we return 0 instead of
    failing the whole pipeline.
    """
    with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f, "html.parser")

    table_tag = None
    for b in soup.find_all("b"):
        if "Environment:WarmupDays" in b.text:
            table_tag = b.find_next("table")
            break
    if table_tag is None:
        return 0

    # Read without trusting header detection; promote first row to header if needed
    df = pd.read_html(StringIO(str(table_tag)), header=None)[0]
    # If columns look generic (e.g., '0', '1'), use first row as header
    generic_cols = all(str(c).isdigit() for c in df.columns)
    if generic_cols and len(df) > 0:
        new_cols = [str(c).strip() for c in df.iloc[0].tolist()]
        df = df.iloc[1:].reset_index(drop=True)
        df.columns = new_cols
    else:
        df.columns = [str(c).strip() for c in df.columns]

    # Normalize and find the warmup column
    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())

    colmap = {norm(c): c for c in df.columns}
    target = None
    for key in ("numberofwarmupdays", "warmupdays"):
        if key in colmap:
            target = colmap[key]
            break
    if target is None:
        # Fallback: any column mentioning both warmup and days
        for c in df.columns:
            nc = norm(c)
            if "warmup" in nc and "day" in nc:
                target = c
                break
    if target is None:
        return 0

    # Take the first numeric value in that column
    def to_float(x):
        try:
            return float(str(x).strip())
        except Exception:
            return None

    series = df[target].map(to_float).dropna()

    if series.empty:
        return 0
    warmup_days = int(series.iloc[0])
    return warmup_days
