from load import load_and_resample_uci_txt, fetch_meteostat_hourly_paris_range
from process import make_features, split_train_test, export_frames
from config import TRAIN_START, TEST_YEAR, RESULTS_DIR

def validate_hourly(df, name):
    assert df.index.is_monotonic_increasing
    assert (df.index.minute == 0).all()
    print(f"{name}: OK ({len(df)} rows)")

if __name__ == "__main__":
    print("Resampling UCI TXT to hourly...")
    df_hourly = load_and_resample_uci_txt(to_csv=True)
    validate_hourly(df_hourly, "power_hourly")
    print("UCI hourly shape:", df_hourly.shape)

    print("Fetching Meteostat hourly weather for 2006–2010 (cached)...")
    full_end = f"{TEST_YEAR}-12-31 23:59:59"
    wx_full = fetch_meteostat_hourly_paris_range(TRAIN_START, full_end, cache=True)
    validate_hourly(wx_full, "weather_hourly")
    print("Meteostat hourly shape:", wx_full.shape)

    print("Building features...")
    feats = make_features(df_hourly, wx_full)
    print("Features shape:", feats.shape)

    X_train, y_train, X_test, y_test = split_train_test(feats)
    print("Train:", X_train.shape, y_train.shape, "| Test:", X_test.shape, y_test.shape)

    export_frames(feats, X_train, y_train, X_test, y_test)
    print("Saved to results/. Done.")
