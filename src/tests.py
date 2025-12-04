"""
Basic data-loading tests for the final project.
These tests DO NOT fetch new data — they only check the pipeline.
"""

from pathlib import Path
import pandas as pd

from load import (
    ensure_uci_txt,
    load_and_resample_uci_txt,
    fetch_meteostat_hourly_paris_range,
)

from config import (
    UCI_TXT,
    HOURLY_CSV,
    WX_FULL_CSV,
)


# -------------------------------------------------------
# Test 1 — UCI TXT file should exist after ensure_uci_txt
# -------------------------------------------------------
def test_ensure_uci_txt():
    ensure_uci_txt()
    assert UCI_TXT.exists(), f"UCI TXT file missing: {UCI_TXT}"


# -------------------------------------------------------------------
# Test 2 — Hourly resampled file should exist & contain a datetime index
# -------------------------------------------------------------------
def test_load_and_resample_uci_txt():
    df = load_and_resample_uci_txt(to_csv=True)

    assert isinstance(df.index, pd.DatetimeIndex), "Index must be DatetimeIndex"
    assert len(df) > 100, "Resampled hourly dataframe seems too small"
    assert HOURLY_CSV.exists(), f"Hourly CSV missing: {HOURLY_CSV}"


# ---------------------------------------------------------
# Test 3 — Weather CSV should exist and contain expected cols
# ---------------------------------------------------------
def test_fetch_meteostat_hourly_paris_range():
    # DO NOT fetch remotely — just load the existing cached file
    assert WX_FULL_CSV.exists(), (
        f"Weather CSV missing: {WX_FULL_CSV}. "
        f"You must run 01_main.py before running tests."
    )

    df_wx = pd.read_csv(WX_FULL_CSV, parse_dates=["datetime"]).set_index("datetime")

    assert isinstance(df_wx.index, pd.DatetimeIndex), "Weather dataframe index must be DatetimeIndex"
    assert len(df_wx) > 100, "Weather dataframe seems too small"

    expected = {"temp_c", "rel_humidity", "wind_speed"}
    present = expected.intersection(df_wx.columns)

    assert present, (
        "Weather dataframe should contain at least one of: "
        "temp_c, rel_humidity, wind_speed"
    )


# ----------------------
# Run all tests manually
# ----------------------
def main():
    print("\n=== Running basic data-loading tests ===\n")

    print("Running test_ensure_uci_txt() ...")
    test_ensure_uci_txt()
    print("✓ ensure_uci_txt() passed\n")

    print("Running test_load_and_resample_uci_txt() ...")
    test_load_and_resample_uci_txt()
    print("✓ load_and_resample_uci_txt() passed\n")

    print("Running test_fetch_meteostat_hourly_paris_range() ...")
    test_fetch_meteostat_hourly_paris_range()
    print("✓ fetch_meteostat_hourly_paris_range() passed\n")

    print("\nAll tests passed! ✔️\n")


if __name__ == "__main__":
    main()
