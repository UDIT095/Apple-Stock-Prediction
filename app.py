import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta, date
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# --- Streamlit Configuration ---
st.set_page_config(page_title="Stock Price Forecast", layout="wide")

# --- Constants ---
MODEL_PATH = 'best_model.pkl'
SCALER_Y_PATH = 'scaler_y.pkl'
SCALER_X_PATH = 'scaler_X_sarimax.pkl'
PREDICTION_HORIZON = 30

# --- US Holidays (2025 sample) ---
US_STOCK_HOLIDAYS_2025 = [
    date(2025, 1, 1), date(2025, 1, 20), date(2025, 2, 17), date(2025, 4, 18),
    date(2025, 5, 26), date(2025, 6, 19), date(2025, 7, 4), date(2025, 9, 1),
    date(2025, 11, 27), date(2025, 12, 25)
]

# --- Load Model & Scalers ---
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)
    try:
        scaler_x = joblib.load(SCALER_X_PATH)
    except FileNotFoundError:
        scaler_x = None
    return model, scaler_y, scaler_x

model, scaler_y, scaler_x = load_artifacts()

is_sarimax = isinstance(model, SARIMAX) or hasattr(model, 'model') and isinstance(model.model, SARIMAX)

# --- UI Section ---
st.title("📈 Stock Price Forecast")
input_method = st.radio("Select Input Method", ["Upload CSV", "Manual Close Price"])

recent_data = None

if input_method == "Upload CSV":
    file = st.file_uploader("Upload recent stock CSV", type="csv")
    if file:
        df = pd.read_csv(file, index_col="Date", parse_dates=True)
        df.sort_index(inplace=True)
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required):
            st.error(f"CSV must include: {required}")
            st.stop()
        recent_data = df
        st.write("📄 Uploaded Data Preview", df.tail())

else:
    close_price = st.number_input("Enter Last Close Price", min_value=1.0, value=150.0)
    today = pd.to_datetime('today').normalize()
    recent_data = pd.DataFrame({'Close': [close_price]}, index=[today])

st.markdown("---")
horizon = st.slider("Forecast Horizon (days)", 1, 60, PREDICTION_HORIZON)
predict_btn = st.button("🔮 Predict")

# --- Simulate Future Dates ---
def generate_future_dates(start, days):
    future = []
    current = start
    while len(future) < days:
        current += timedelta(days=1)
        if current.weekday() < 5 and current not in US_STOCK_HOLIDAYS_2025:
            future.append(pd.to_datetime(current))
    return future

# --- Simulate Smoothed Exogenous Variables ---
def simulate_future_exog(df, steps, scaler_x):
    window = 10
    rolling = df[['Open', 'High', 'Low', 'Volume', 'Close']].rolling(window=window).mean().dropna()
    base = rolling.iloc[-1]

    sim_data = {
        'Open': np.random.normal(base['Open'], base['Open'] * 0.005, steps),
        'High': np.random.normal(base['High'], base['High'] * 0.005, steps),
        'Low': np.random.normal(base['Low'], base['Low'] * 0.005, steps),
        'Volume': np.random.normal(base['Volume'], base['Volume'] * 0.05, steps),
        'Close_Lag_30': np.linspace(df['Close'].iloc[-1], df['Close'].iloc[-1] * 1.02, steps),
        'Daily_Return': np.random.normal(0, 0.3, steps)
    }

    future_exog = pd.DataFrame(sim_data)
    return pd.DataFrame(scaler_x.transform(future_exog), columns=future_exog.columns)

# --- Prediction ---
if predict_btn and recent_data is not None:
    st.subheader("📊 Forecast Output")
    last_date = recent_data.index[-1].date()
    future_dates = generate_future_dates(last_date, horizon)

    try:
        if is_sarimax and scaler_x is not None and input_method == "Upload CSV":
            future_exog_scaled = simulate_future_exog(recent_data, len(future_dates), scaler_x)
            if hasattr(model, "forecast"):
                forecast = model.forecast(steps=len(future_dates), exog=future_exog_scaled)
            else:
                forecast = model.predict(n_periods=len(future_dates), X=future_exog_scaled)
        else:
            forecast = model.predict(n_periods=len(future_dates))

        predictions_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted Close Price": forecast
        }).set_index("Date")

        st.line_chart(predictions_df)

        st.dataframe(predictions_df.style.format({"Predicted Close Price": "${:.2f}"}))

        # Plot historical vs forecast
        fig, ax = plt.subplots(figsize=(12, 6))
        if 'Close' in recent_data.columns:
            ax.plot(recent_data.index[-60:], recent_data['Close'].iloc[-60:], label="Historical Close", color='blue')
        else:
            ax.plot(recent_data.index, recent_data['Close'], label="Last Known Close", color='blue')

        ax.plot(predictions_df.index, predictions_df['Predicted Close Price'], label="Forecast", color='red', linestyle='--')
        ax.set_title("Stock Price Forecast")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Prediction Error: {e}")

st.markdown("---")
st.info("This forecast uses pre-trained ARIMA/SARIMAX/XGBoost models. For best results, ensure the model and scalers are correctly saved and match your training data features.")
