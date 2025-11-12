import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# ARIMA -> SARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{name:10s} | MAE={mae:.3f} | RMSE={rmse:.3f}")
    return mae, rmse


def run_models(
    train_path: Path,
    test_path: Path,
    show_hours: int | None = 200,
    save_dir: Path | None = None,
):
    # load data
    train_df = pd.read_csv(train_path, index_col=0, parse_dates=True)
    test_df = pd.read_csv(test_path, index_col=0, parse_dates=True)

    y_train = train_df["load"]
    y_test = test_df["load"]
    feature_cols = [c for c in train_df.columns if c != "load"]

    print(f"Train samples: {len(y_train)} | Test samples: {len(y_test)}")

    results = {}
    preds = {}

    # -------------------- SARIMA --------------------
    # 24-hour seasonality; feel free to tune (p,d,q)x(P,D,Q,24)
    print("\nTraining SARIMA (2,1,2) x (1,1,1,24)...")
    sarima = SARIMAX(
        y_train,
        order=(2, 1, 2),
        seasonal_order=(1, 1, 1, 24),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    y_pred_sarima = sarima.forecast(steps=len(y_test))
    preds["SARIMA"] = pd.Series(np.asarray(y_pred_sarima).ravel(), index=y_test.index)
    results["SARIMA"] = evaluate(y_test, preds["SARIMA"], "SARIMA")

    # --------------------- ETS ---------------------
    print("\nTraining ETS (trend=add, seasonal=add, period=24)...")
    ets = ExponentialSmoothing(y_train, trend="add", seasonal="add", seasonal_periods=24).fit()
    y_pred_ets = ets.forecast(len(y_test))
    preds["ETS"] = pd.Series(np.asarray(y_pred_ets).ravel(), index=y_test.index)
    results["ETS"] = evaluate(y_test, preds["ETS"], "ETS")

    # ------------------- XGBoost -------------------
    print("\nTraining XGBoost...")
    X_train, X_test = train_df[feature_cols], test_df[feature_cols]
    xgb = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    preds["XGBoost"] = pd.Series(np.asarray(y_pred_xgb).ravel(), index=y_test.index)
    results["XGBoost"] = evaluate(y_test, preds["XGBoost"], "XGBoost")

    # --------------------- LSTM --------------------
    print("\nTraining LSTM...")
    X_train_lstm = np.expand_dims(X_train.values, axis=1)  # [samples, timesteps=1, features]
    X_test_lstm = np.expand_dims(X_test.values, axis=1)

    lstm = Sequential([
        LSTM(64, input_shape=(1, X_train.shape[1])),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    lstm.compile(optimizer="adam", loss="mse")
    lstm.fit(X_train_lstm, y_train.values, epochs=10, batch_size=32, verbose=0)

    y_pred_lstm = lstm.predict(X_test_lstm, verbose=0).flatten()
    preds["LSTM"] = pd.Series(np.asarray(y_pred_lstm).ravel(), index=y_test.index)
    results["LSTM"] = evaluate(y_test, preds["LSTM"], "LSTM")

    # ----------------- aggregate & save -----------------
    comp = pd.DataFrame(
        {
            "Actual": y_test,
            "SARIMA": preds["SARIMA"],
            "ETS": preds["ETS"],
            "XGBoost": preds["XGBoost"],
            "LSTM": preds["LSTM"],
        },
        index=y_test.index,
    )

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        comp.to_csv(save_dir / "predictions_test.csv")
        pd.DataFrame(results, index=["MAE", "RMSE"]).T.to_csv(save_dir / "model_results.csv")
        print(f"\nSaved: {save_dir / 'predictions_test.csv'}")
        print(f"Saved: {save_dir / 'model_results.csv'}")

    # --------------------- plotting ---------------------
    if (show_hours is None) or (show_hours >= len(comp)):
        view = comp
        title_suffix = "(full test set)"
    else:
        view = comp.iloc[:show_hours]
        title_suffix = f"(first {show_hours} hours)"

    plt.figure(figsize=(12, 6))
    plt.plot(view.index, view["Actual"], label="Actual", color="black")
    plt.plot(view.index, view["SARIMA"], label="SARIMA")
    plt.plot(view.index, view["ETS"], label="ETS")
    plt.plot(view.index, view["XGBoost"], label="XGBoost")
    plt.plot(view.index, view["LSTM"], label="LSTM")
    plt.legend()
    plt.title(f"Load Forecasting: {title_suffix}")
    plt.tight_layout()

    if save_dir is not None:
        fig_path = save_dir / "comparison_plot.png"
        plt.savefig(fig_path, dpi=150)
        print(f"Saved: {fig_path}")

    plt.show()


if __name__ == "__main__":
    results_dir = Path("../results")
    run_models(
        train_path=results_dir / "train_features.csv",
        test_path=results_dir / "test_features.csv",
        show_hours=200,          # set None for the full test set
        save_dir=results_dir,    # set None to skip saving files
    )
