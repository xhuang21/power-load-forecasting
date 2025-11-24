# ⚡ Power Load Forecasting

A comparative study of **SARIMA**, **ETS**, **XGBoost**, and **LSTM** models for short-term power load forecasting using historical consumption and weather data.

---

## 📘 Project Overview
This project focuses on predicting hourly power load based on past consumption patterns and meteorological factors such as temperature, humidity, and wind speed.  
The dataset spans **2006–2010** and is resampled to hourly intervals for consistent model input.

---

## 🌍 Data Sources

This project uses two external data sources:

---

### **1. UCI Household Power Consumption Dataset (Auto-Download)**

The script automatically downloads the dataset from the UCI Machine Learning Repository:

Official dataset page:
https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

Direct ZIP download link used by the project:
https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip

The project automatically performs the following steps:

- Downloads `household_power_consumption.zip` into the `data/` directory  
- Extracts `household_power_consumption.txt`  
- Converts and resamples it into hourly data  
- Saves the processed file as `power_hourly_2006_2010.csv`

**No manual download is required.**

---

### **2. Meteostat Weather API (Auto-Fetch)**

Hourly weather data for **Paris** are retrieved automatically using the `meteostat` API.

The weather dataset includes:

- Temperature (`temp_c`)
- Relative humidity (`rel_humidity`)
- Wind speed (`wind_speed`)

The project automatically performs the following steps:

- Fetches hourly weather data (2006–2010) from the Meteostat API  
- Cleans and standardizes the variables  
- Saves the processed file as `meteostat_paris_2006_2010.csv` in the `data/` directory  
- Uses local cache to avoid repeated API calls

**No manual download from Meteostat is required.**

---


---

## ⚙️ Project Structure

```
power-load-forecasting/
├power-load-forecasting/
├── data/ # Raw and processed datasets ⚠️ ignored by .gitignore
│ 
│
├── doc/                          # Final report and presentation slides
│   ├── final_project_report.pdf
│   └── Final_Project_Presentation.pdf
│
├── results/ # Model results and comparison plots ⚠️ ignored by .gitignore
│ 
│
├── src/                          # Source code modules
│   ├── 01_main.py                # Main execution script: loads data, trains models
│   ├── 02_analyze.py             # Model evaluation and metric computation
│   ├── 03_plot_results.py        # Generates RMSE/MAE bar charts from results CSV
│   ├── 04_plot_scatter.py        # Generates scatter plots (Actual vs Predicted)
│   ├── config.py                 # Global configuration settings
│   ├── load.py                   # Load & preprocess UCI + Meteostat datasets
│   ├── process.py                # Feature engineering and dataset preparation
│   └── tests.py                  # Basic sanity checks (optional)
│
│
├── .gitignore # Ignore temp files, caches, data/results
├── requirements.txt # Dependencies
├── README.md # Project documentation
└── LICENSE # MIT License
```

---

## 🧠 Models Implemented

| Model     | Description                                         | Library       |
|------------|----------------------------------------------------|----------------|
| **SARIMA** | Seasonal ARIMA for capturing trend and seasonality | `statsmodels` |
| **ETS**    | Error-Trend-Seasonal exponential smoothing model   | `statsmodels` |
| **XGBoost**| Gradient boosting for tabular regression           | `xgboost`     |
| **LSTM**   | Recurrent neural network for sequential forecasting| `tensorflow`  |

---

## 🚀 How to Run the Project

### 1. Clone the repository
```bash
git clone <your_repo_url>
cd <repo_folder>
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 3. Run preprocessing & feature generation  
This step loads raw UCI + Meteostat data and produces merged hourly datasets and model-ready features.

```bash
python src/01_main.py
```

Output (stored in `results/`):
- `train_features.csv`
- `test_features.csv`

---

## 4. Train and evaluate all models  
This script trains SARIMA, ETS, XGBoost, and LSTM, and produces 200-hour predictions + metrics.

```bash
python src/02_analyze.py
```

Output (stored in `results/`):
- `model_results.csv`  
- `predictions_test.csv`

---

## 5. Generate model performance bar chart (MAE & RMSE)

```bash
python src/03_plot_results.py
```

Output saved to:

```
results/model_error_bar_200h.png
```

---

## 6. Generate individual scatter plots (Actual vs Predicted) for all 4 models  
This script creates four separate scatter plots based on the first 200 hours.

```bash
python src/04_plot_scatter.py
```

Outputs saved to:

```
results/scatter_sarima.png
results/scatter_ets.png
results/scatter_xgboost.png
results/scatter_lstm.png
```

---

## ✔ All results are saved under:

```
results/
```

No data is uploaded to GitHub (per course rules).  
All plots and metrics are generated automatically by running the above scripts.



## 📊 Key Insights

- Statistical models (**SARIMA**, **ETS**) perform reliably on stable seasonal data.  
- Machine learning and deep learning (**XGBoost**, **LSTM**) excel with nonlinear dependencies.  
- Proper feature engineering (lags, rolling means, weather factors) boosts performance.  
- **LSTM** achieves the highest accuracy across all evaluation metrics.

---

## 🔮 Future Work

- Extend dataset with post-2010 years  
- Integrate real-time weather forecasts  
- Explore hybrid models (LSTM + XGBoost)  
- Deploy as an API for live energy forecasting  

---

## 👨‍💻 Author

Developed by **Xinyuan Huang**  
*Final Project — Power Load Forecasting (2025)*  

---

## ⚖️ License

This project is released under the **MIT License**.
