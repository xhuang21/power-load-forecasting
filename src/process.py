import pandas as pd
from config import TRAIN_START, TRAIN_END, TEST_YEAR, RESULTS_DIR

def make_features(power_hourly: pd.DataFrame, weather_hourly: pd.DataFrame) -> pd.DataFrame:
    df = power_hourly.copy()
    df = df[["Global_active_power"]].rename(columns={"Global_active_power": "load"})
    df = df.join(weather_hourly, how="left")

    df["hour"] = df.index.hour
    df["dow"] = df.index.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    for h in [1, 2, 24]:
        df[f"lag_{h}h"] = df["load"].shift(h)
    for w in [3, 24]:
        df[f"roll_mean_{w}h"] = df["load"].rolling(window=w, min_periods=w).mean()

    df = df.dropna()
    return df

def split_train_test(df: pd.DataFrame):
    train = df.loc[TRAIN_START:TRAIN_END].copy()
    test = df.loc[f"{TEST_YEAR}-01-01":f"{TEST_YEAR}-12-31 23:59:59"].copy()
    X_train, y_train = train.drop(columns=["load"]), train["load"]
    X_test, y_test = test.drop(columns=["load"]), test["load"]
    return X_train, y_train, X_test, y_test

def export_frames(df: pd.DataFrame, X_train, y_train, X_test, y_test):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "features_full.csv")
    pd.concat([y_train, X_train], axis=1).to_csv(RESULTS_DIR / "train_features.csv")
    pd.concat([y_test, X_test], axis=1).to_csv(RESULTS_DIR / "test_features.csv")
