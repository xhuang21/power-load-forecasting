import io
import zipfile
from pathlib import Path
from datetime import datetime

import pandas as pd
import requests
from meteostat import Point, Hourly

from config import (
    DATA_DIR,
    UCI_TXT,
    HOURLY_CSV,
    WX_FULL_CSV,
    UCI_ZIP,
    UCI_ZIP_URL,
    PARIS_LAT,
    PARIS_LON,
    TEST_YEAR,
)


def ensure_uci_txt() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if UCI_TXT.exists():
        return UCI_TXT

    resp = requests.get(UCI_ZIP_URL, stream=True, timeout=90)
    resp.raise_for_status()
    zip_bytes = resp.content

    try:
        with open(UCI_ZIP, "wb") as f:
            f.write(zip_bytes)
    except OSError:
        pass

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        member = next((n for n in zf.namelist() if n.endswith(".txt")), None)
        if member is None:
            raise RuntimeError("No .txt file found in UCI zip archive")
        with zf.open(member) as src, open(UCI_TXT, "wb") as out:
            out.write(src.read())

    if not UCI_TXT.exists():
        raise RuntimeError("Failed to create UCI txt file")

    return UCI_TXT


def load_and_resample_uci_txt(to_csv: bool = True) -> pd.DataFrame:
    ensure_uci_txt()

    df = pd.read_csv(
        UCI_TXT,
        sep=";",
        na_values="?",
        parse_dates={"datetime": ["Date", "Time"]},
        dayfirst=True,
        low_memory=False,
        encoding="latin-1",
    )
    df = df.set_index("datetime").sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    df_hourly = df.resample("h").mean()

    if to_csv:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df_hourly.to_csv(HOURLY_CSV, index=True)

    return df_hourly


def fetch_meteostat_hourly_paris_range(
    start_ts: str,
    end_ts: str,
    cache: bool = True,
) -> pd.DataFrame:
    if cache and WX_FULL_CSV.exists():
        wx = pd.read_csv(WX_FULL_CSV, parse_dates=["datetime"]).set_index("datetime")
        return wx

    start = pd.to_datetime(start_ts)
    end = pd.to_datetime(end_ts)
    loc = Point(PARIS_LAT, PARIS_LON)
    wx = Hourly(loc, start, end).fetch()

    rename_map = {
        "t": "temp_c",
        "r": "rel_humidity",
        "w": "wind_speed",
        "temp": "temp_c",
        "rhum": "rel_humidity",
        "wspd": "wind_speed",
    }
    wx = wx.rename(columns=rename_map)
    keep = [c for c in ["temp_c", "rel_humidity", "wind_speed"] if c in wx.columns]
    out = wx[keep].copy() if keep else wx.copy()
    out.index.name = "datetime"

    if cache:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        out.to_csv(WX_FULL_CSV, index=True)

    return out
