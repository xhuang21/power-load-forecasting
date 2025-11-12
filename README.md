# ⚡ Power Load Forecasting

A comparative study of **SARIMA**, **ETS**, **XGBoost**, and **LSTM** models for short-term power load forecasting using historical consumption and weather data.

---

## 📘 Project Overview
This project focuses on predicting hourly power load based on past consumption patterns and meteorological factors such as temperature, humidity, and wind speed.  
The dataset spans **2006–2010** and is resampled to hourly intervals for consistent model input.

---

## 🌍 Data Sources

- **Household Power Consumption:** UCI Machine Learning Repository  
  Original dataset: `household_power_consumption.txt`

- **Weather Data:** Retrieved from Meteostat API for **Paris**, including:  
  - Temperature (°C)  
  - Relative humidity (%)  
  - Wind speed (m/s)

All data is merged and processed into `power_hourly_2006_2010.csv` under the `/data` directory.

---

## ⚙️ Project Structure

```
power-load-forecasting/
├── data/                     # Raw and processed datasets
│   ├── household_power_consumption.txt
│   ├── meteostat_paris_2006_2010.csv
│   └── power_hourly_2006_2010.csv
│
├── results/                  # Generated model results and comparison plots
│   ├── comparison_plot.png
│   ├── model_results.csv
│   ├── predictions_test.csv
│   ├── test_features.csv
│   ├── train_features.csv
│   └── features_full.csv
│
├── src/                      # Source code modules
│   ├── load.py               # Data loading utilities
│   ├── process.py            # Data preprocessing & feature engineering
│   ├── analyze.py            # Model training and evaluation
│   ├── config.py             # Global configuration
│   └── main.py               # Main script entry point
│
├── tests/                    # Unit tests (optional)
├── requirements.txt          # Dependencies
├── README.md                 # Project documentation
└── LICENSE
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
    git clone https://github.com/<your-username>/power-load-forecasting.git
    cd power-load-forecasting
    ```

2. **Install dependencies**

    ```bash
    pip install -r requirements.txt
    ```

3. **Run preprocessing and feature generation**

    ```bash
    python src/process.py
    ```

4. **Train and evaluate all models**

    ```bash
    python src/analyze.py
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
