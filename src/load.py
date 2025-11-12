import pandas as pd
from datetime import datetime
from meteostat import Point, Hourly
from config import (
    UCI_TXT, HOURLY_CSV, DATA_DIR,
    PARIS_LAT, PARIS_LON, TEST_YEAR, WX_FULL_CSV
)

def load_and_resample_uci_txt(to_csv: bool = True) -> pd.DataFrame:
    df = pd.read_csv(
        UCI_TXT,
        sep=';',
        na_values='?',
        parse_dates={'datetime': ['Date', 'Time']},
        dayfirst=True,
        low_memory=False,
        encoding='latin-1'
    )
    df = df.set_index('datetime').sort_index()
    df = df.apply(pd.to_numeric, errors='coerce')
    df_hourly = df.resample('h').mean()
    if to_csv:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df_hourly.to_csv(HOURLY_CSV, index=True)
    return df_hourly

def fetch_meteostat_hourly_paris_range(start_ts: str, end_ts: str, cache: bool = True) -> pd.DataFrame:
    if cache and WX_FULL_CSV.exists():
        wx = pd.read_csv(WX_FULL_CSV, parse_dates=["datetime"]).set_index("datetime")
        return wx

    start = pd.to_datetime(start_ts)
    end = pd.to_datetime(end_ts)
    loc = Point(PARIS_LAT, PARIS_LON)
    wx = Hourly(loc, start, end).fetch()

    rename_map = {
        't': 'temp_c', 'r': 'rel_humidity', 'w': 'wind_speed',
        'temp': 'temp_c', 'rhum': 'rel_humidity', 'wspd': 'wind_speed'
    }
    wx = wx.rename(columns=rename_map)
    keep = [c for c in ['temp_c', 'rel_humidity', 'wind_speed'] if c in wx.columns]
    out = wx[keep].copy() if keep else wx.copy()
    out.index.name = 'datetime'

    if cache:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        out.to_csv(WX_FULL_CSV, index=True)
    return out
