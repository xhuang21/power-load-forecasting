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
│ ├── household_power_consumption.txt
│ ├── meteostat_paris_2006_2010.csv
│ └── power_hourly_2006_2010.csv
│
├── doc/ # Documentation and final report
│ └── final_project_report.pdf
│
├── results/ # Model results and comparison plots ⚠️ ignored by .gitignore
│ ├── comparison_plot.png
│ ├── model_results.csv
│ ├── predictions_test.csv
│ ├── test_features.csv
│ ├── train_features.csv
│ └── features_full.csv
│
├── src/ # Source code modules
│ ├── load.py # Load and preprocess UCI + Meteostat data
│ ├── process.py # Feature engineering and dataset creation
│ ├── analyze.py # Model training and evaluation
│ ├── config.py # Global configuration
│ ├── main.py # Main execution entry point
│ └── tests.py # Basic sanity checks (optional)
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

## 🚀 How to Run

1. **Clone the repository**

    ```bash
    git clone https://github.com/<xhuang21>/power-load-forecasting.git
    cd power-load-forecasting
    ```

2. **Install dependencies**

    ```bash
    pip install -r requirements.txt
    ```

3. **Run preprocessing and feature generation**

    ```bash
    python src/01_main.py
    ```

4. **Train and evaluate all models**

    ```bash
    python src/02_analyze.py
    ```

5. **Check results**

    - Model metrics → `results/model_results.csv`  
    - Forecast comparison plot → `results/comparison_plot.png`

---

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
