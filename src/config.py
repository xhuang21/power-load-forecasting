from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

UCI_TXT = DATA_DIR / "household_power_consumption.txt"
HOURLY_CSV = DATA_DIR / "power_hourly_2006_2010.csv"
WX_FULL_CSV = DATA_DIR / "meteostat_paris_2006_2010.csv"

TRAIN_START = "2006-01-01"
TRAIN_END   = "2009-12-31 23:59:59"
TEST_YEAR   = "2010"

PARIS_LAT, PARIS_LON = 48.8566, 2.3522
