import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib

# ===============================
# Load Data
# ===============================
df = pd.read_csv("hour.csv")

# ===============================
# Load Scalers
# ===============================
X_scaler = joblib.load("x_scaler.pkl")
y_scaler = joblib.load("y_scaler.pkl")

# ===============================
# Positional Encoding
# ===============================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# ===============================
# Model
# ===============================
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=3):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.2,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x[:, -1, :]
        x = self.norm(x)
        return self.fc(x).squeeze()

# ===============================
# Load Model
# ===============================
input_dim = X_scaler.n_features_in_

model = TransformerModel(input_dim)
model.load_state_dict(torch.load("transformer_model.pth", map_location="cpu"))
model.eval()

# ===============================
# Sidebar
# ===============================
st.sidebar.title("🚲 Bike Demand App")
page = st.sidebar.radio("Go to", ["Dashboard", "Prediction", "About"])

# ===============================
# Dashboard
# ===============================
if page == "Dashboard":
    st.title("📊 Bike Demand Dashboard")

    if "hr" in df.columns:
        st.line_chart(df.groupby("hr")["cnt"].mean())

    if "season" in df.columns:
        st.bar_chart(df.groupby("season")["cnt"].mean())

    if "weathersit" in df.columns:
        st.bar_chart(df.groupby("weathersit")["cnt"].mean())

# ===============================
# Prediction (PERFECT VERSION)
# ===============================
elif page == "Prediction":
    st.title("🔮 Bike Demand Prediction")

    # Inputs
    season = st.selectbox("Season", [1,2,3,4])
    yr = st.selectbox("Year (0=2011,1=2012)", [0,1])
    mnth = st.slider("Month", 1, 12, 6)
    holiday = st.selectbox("Holiday", [0,1])
    workingday = st.selectbox("Working Day", [0,1])
    weathersit = st.selectbox("Weather", [1,2,3,4])

    temp = st.slider("Temperature", 0.0, 1.0, 0.5)
    atemp = st.slider("Feels Like Temp", 0.0, 1.0, 0.5)
    hum = st.slider("Humidity", 0.0, 1.0, 0.5)
    windspeed = st.slider("Windspeed", 0.0, 1.0, 0.2)

    hour = st.slider("Hour", 0, 23, 12)
    weekday = st.slider("Weekday", 0, 6, 3)

    if st.button("Predict"):

        # Cyclical features
        hour_sin = np.sin(2*np.pi*hour/24)
        hour_cos = np.cos(2*np.pi*hour/24)

        weekday_sin = np.sin(2*np.pi*weekday/7)
        weekday_cos = np.cos(2*np.pi*weekday/7)

        # FULL feature vector (must match training)
        features = np.array([[
            season, yr, mnth, holiday, workingday, weathersit,
            temp, atemp, hum, windspeed,
            hour_sin, hour_cos, weekday_sin, weekday_cos
        ]])

        # Scale
        input_scaled = X_scaler.transform(features)

        # Sequence
        seq = np.repeat(input_scaled, 24, axis=0)
        seq = seq.reshape(1, 24, -1)

        input_tensor = torch.tensor(seq, dtype=torch.float32)

        with torch.no_grad():
            pred = model(input_tensor).numpy()

        pred_original = y_scaler.inverse_transform(pred.reshape(-1,1))[0][0]

        st.success(f"Predicted Bike Demand: {pred_original:.2f}")

        if pred_original < 100:
            st.error("⚠️ Low Availability")
        else:
            st.success("✅ Good Availability")

# ===============================
# About
# ===============================
elif page == "About":
    st.title("ℹ️ About")

    st.markdown("""
    - Transformer-based Time Series Model  
    - Sequence Length: 24 hours  
    - R² Score: ~0.92  

    This app predicts bike demand using weather and time features.
    """)
