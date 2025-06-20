# 📈 Apple Stock Price Forecasting App

This project aims to forecast Apple Inc.'s stock price for the next 30 days using advanced time series forecasting techniques such as **ARIMA**, **SARIMAX**, and **XGBoost**. The project includes data preprocessing, feature engineering, model evaluation, and a deployed interactive **Streamlit web app**.

---

## 📌 Project Objectives

- Predict Apple’s stock prices for the next 30 days using historical data.
- Build and evaluate statistical and machine learning models.
- Analyze seasonality, volatility, and external impacts on stock prices.
- Deploy a user-friendly web application for real-time predictions.

---

## 🧠 Models Used

- **ARIMA / SARIMAX**: Classical statistical models for trend and seasonality.
- **XGBoost**: Machine learning model for regression-based time series.
- **Model Comparison**: RMSE and MAE are used to select the best-performing model.

---

## 📊 Dataset

The dataset includes Apple stock data from **2012 to 2019**:

| Column | Description |
|--------|-------------|
| `Date` | Trading date |
| `Open` | Opening price |
| `High` | Highest price of the day |
| `Low`  | Lowest price of the day |
| `Close`| Closing price |
| `Volume` | Shares traded |

**Target**: Next 30-Day Close Price Forecast

---

## 🚀 Streamlit Web App

The app provides:
- Options to **upload a CSV** file or manually enter the last close price.
- Dynamic **forecast horizon selection** (1–60 days).
- Interactive **line charts** comparing historical and predicted stock prices.
- **Holiday-aware** future date generation for realistic trading day forecasts.
- Handles both ARIMA/SARIMAX and machine learning models automatically.

### Run the App

```bash
streamlit run app.py
```

> Ensure that the following files are in the same directory:
> - `best_model.pkl` (trained model)
> - `scaler_y.pkl` (target scaler)
> - `scaler_X_sarimax.pkl` (exogenous feature scaler, if using SARIMAX)

---

## ⚙️ Project Pipeline

1. **Data Preprocessing**: Handling missing values, scaling, and lag features.
2. **Exploratory Data Analysis**: ACF, PACF, box plots, seasonal decomposition.
3. **Model Training**: ARIMA/SARIMAX/XGBoost with hyperparameter tuning.
4. **Evaluation**: Metrics like RMSE, MAE, and R².
5. **Forecasting**: Generating future values with/without exogenous variables.
6. **Deployment**: Streamlit app with prediction and visualization.

---

## 🧪 Requirements

```bash
pip install -r requirements.txt
```

Typical dependencies include:
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn`, `xgboost`
- `statsmodels`
- `streamlit`, `joblib`

---

## 📁 File Structure

```
├── app.py                     # Streamlit app (rename .txt to .py)
├── notebook.ipynb             # Jupyter notebook for model development
├── best_model.pkl             # Trained model file
├── scaler_y.pkl               # Target scaler
├── scaler_X_sarimax.pkl       # Feature scaler (for SARIMAX)
├── README.md                  # Project documentation
├── data.csv                   # Historical Apple stock data (2012–2019)
```

---

## 📬 Contact

For questions or feedback, feel free to reach out via GitHub or email.
