import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ------------------------------
# 1. Title & Intro
# ------------------------------
st.title(" P675 - Apple Stock Price Prediction (Next 30 Days)")
st.write("This app uses a trained XGBoost model to forecast Apple stock prices.")

# ------------------------------
# 2. Load Model & Data
# ------------------------------
model = joblib.load("xgb_model.pkl")
data = pd.read_csv("P675 DATASET.csv")
data['Date'] = pd.to_datetime(data['Date'])

# ------------------------------
# 3. Slider for Forecast Horizon
# ------------------------------
days = st.slider("Select forecast horizon (days)", 1, 30, 10)

# ------------------------------
# 4. Forecast Logic
# ------------------------------
history = list(data['Close'].values)
future_preds = []

for i in range(days):
    lag_1 = history[-1]
    lag_7 = history[-7] if len(history) >= 7 else lag_1
    ma_30 = np.mean(history[-30:]) if len(history) >= 30 else np.mean(history)
    vol_30 = np.std(history[-30:]) if len(history) >= 30 else np.std(history)

    next_features = pd.DataFrame([[lag_1, lag_7, ma_30, vol_30]],
                                 columns=['Lag_1','Lag_7','MA_30','Volatility'])
    next_pred = model.predict(next_features)[0]
    future_preds.append(float(next_pred))   # convert np.float32 → float
    history.append(next_pred)

future_dates = pd.date_range(start=data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=days)

# ------------------------------
# 5. Show Forecast Table
# ------------------------------
forecast_df = pd.DataFrame({"Date": future_dates, "Predicted Price ($)": future_preds})
st.subheader(" Predicted Prices (Clean Table)")
st.dataframe(forecast_df)

# ------------------------------
# 6. Forecast Visualization (Last 100 Days + Forecast)
# ------------------------------
st.subheader("📉 Forecast Visualization")
fig, ax = plt.subplots(figsize=(10,5))

# Actual last 100 days
ax.plot(data['Date'].tail(100), data['Close'].tail(100), label='Actual (Last 100 Days)', color='black')

# Forecast
ax.plot(forecast_df["Date"], forecast_df["Predicted Price ($)"], label='Forecast (Next Days)', color='red', marker='o')

ax.set_title("Apple Stock Price Forecast — Actual vs Predicted")
ax.set_xlabel("Date")
ax.set_ylabel("Price ($)")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# ------------------------------
# 7. Final XGBoost Training/Testing Visualization
# ------------------------------
st.subheader("Training, Testing & Predicted Values")
train_size = int(len(data['Close']) * 0.8)

fig2, ax2 = plt.subplots(figsize=(10,5))

# Training data
ax2.plot(data['Date'][:train_size], data['Close'][:train_size], label='Training Data', color='blue')

# Testing data
ax2.plot(data['Date'][train_size:], data['Close'][train_size:], label='Testing Data', color='green')

# Predicted (XGBoost forecast)
ax2.plot(forecast_df["Date"], forecast_df["Predicted Price ($)"], label='Predicted (Next 30 Days)', color='red')

ax2.set_title("Apple Stock Price Forecast — Training, Testing & Predicted Values")
ax2.set_xlabel("Date")
ax2.set_ylabel("Price ($)")
ax2.legend()
ax2.grid(True)

st.pyplot(fig2)
