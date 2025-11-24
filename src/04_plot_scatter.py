from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():


    base_dir = Path(__file__).resolve().parents[1]
    results_dir = base_dir / "results"


    preds_path = results_dir / "predictions_test.csv"

    print(f"Reading prediction file → {preds_path}")


    df = pd.read_csv(preds_path)


    df_200 = df.iloc[:200]


    models = {
        "SARIMA": "scatter_sarima.png",
        "ETS": "scatter_ets.png",
        "XGBoost": "scatter_xgboost.png",
        "LSTM": "scatter_lstm.png"
    }


    for model, filename in models.items():

        plt.figure(figsize=(7, 6))


        plt.scatter(df_200["Actual"], df_200[model], s=12, alpha=0.5)


        min_val = min(df_200["Actual"].min(), df_200[model].min())
        max_val = max(df_200["Actual"].max(), df_200[model].max())
        plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1)

        plt.title(f"Actual vs Predicted ({model}) — First 200 Hours")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")


        save_path = results_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved → {save_path}")


if __name__ == "__main__":
    main()
