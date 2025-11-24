from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():

    base_dir = Path(__file__).resolve().parents[1]
    results_dir = base_dir / "results"

    metrics_path = results_dir / "model_results.csv"

    print("Reading:", metrics_path)


    metrics_df = pd.read_csv(metrics_path)


    first_col = metrics_df.columns[0]
    metrics_df = metrics_df.rename(columns={first_col: "model"})


    order = ["SARIMA", "ETS", "XGBoost", "LSTM"]
    metrics_df = metrics_df.set_index("model").loc[order].reset_index()


    models = metrics_df["model"].tolist()
    rmse_vals = metrics_df["RMSE"].values
    mae_vals = metrics_df["MAE"].values


    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, rmse_vals, width, label="RMSE")
    plt.bar(x + width/2, mae_vals, width, label="MAE")

    plt.xticks(x, models)
    plt.ylabel("Error")
    plt.title("Model Performance (First 200 Hours)")
    plt.legend()
    plt.tight_layout()


    save_path = results_dir / "model_error_bar_200h.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved bar chart → {save_path}")


if __name__ == "__main__":
    main()
