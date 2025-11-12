# optional sanity checks if needed
import pandas as pd
from pathlib import Path

def quick_check(csv_path: Path):
    df = pd.read_csv(csv_path, nrows=10)
    print(df.head())

if __name__ == "__main__":
    quick_check(Path("../data/power_hourly_2006_2010.csv"))
