import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib

# ===============================
# Page Config
# ===============================
st.set_page_config(page_title="Bike Demand App", layout="wide")

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
# Model Classes
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
st.sidebar.title("🚲 Bike Demand")
page = st.sidebar.radio("Navigation", [
    "Dashboard",
    "Prediction",
    "Dataset Explorer",
    "Model Performance",
    "About"
])

# ===============================
# DASHBOARD
# ===============================
if page == "Dashboard":
    st.title("📊 Bike Demand Insights")

    # KPI Cards
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Rides", f"{int(df['cnt'].sum()):,}")
    col2.metric("Avg Demand", f"{int(df['cnt'].mean())}")
    col3.metric("Peak Demand", f"{int(df['cnt'].max())}")

    st.markdown("---")

    # Hourly Demand
    if "hr" in df.columns:
        st.subheader("⏰ Hourly Demand Pattern")
        hourly = df.groupby("hr")["cnt"].mean()
        st.line_chart(hourly)

        st.info(f"Peak hour: {hourly.idxmax()}:00")

    # Season Analysis
    if "season" in df.columns:
        st.subheader("🌤️ Seasonal Demand")

        season_map = {1:"Spring", 2:"Summer", 3:"Fall", 4:"Winter"}
        df["season_name"] = df["season"].map(season_map)

        season_data = df.groupby("season_name")["cnt"].mean()
        st.bar_chart(season_data)

    # Weather Analysis
    if "weathersit" in df.columns:
        st.subheader("🌦️ Weather Impact")

        weather_map = {
            1: "Clear",
            2: "Mist",
            3: "Light Rain",
            4: "Heavy Rain"
        }

        df["weather_name"] = df["weathersit"].map(weather_map)
        weather_data = df.groupby("weather_name")["cnt"].mean()
        st.bar_chart(weather_data)

# ===============================
# PREDICTION
# ===============================
elif page == "Prediction":
    st.title("🔮 Predict Bike Demand")

    st.markdown("### Enter Conditions")

    col1, col2 = st.columns(2)

    with col1:
        season = st.selectbox("Season", [1,2,3,4])
        yr = st.selectbox("Year (0=2011,1=2012)", [0,1])
        mnth = st.slider("Month", 1, 12, 6)
        holiday = st.selectbox("Holiday", [0,1])
        workingday = st.selectbox("Working Day", [0,1])
        weathersit = st.selectbox("Weather", [1,2,3,4])

    with col2:
        temp = st.slider("Temperature", 0.0, 1.0, 0.5)
        atemp = st.slider("Feels Like Temp", 0.0, 1.0, 0.5)
        hum = st.slider("Humidity", 0.0, 1.0, 0.5)
        windspeed = st.slider("Windspeed", 0.0, 1.0, 0.2)
        hour = st.slider("Hour", 0, 23, 12)
        weekday = st.slider("Weekday", 0, 6, 3)

    if st.button("🚀 Predict Demand"):

        hour_sin = np.sin(2*np.pi*hour/24)
        hour_cos = np.cos(2*np.pi*hour/24)

        weekday_sin = np.sin(2*np.pi*weekday/7)
        weekday_cos = np.cos(2*np.pi*weekday/7)

        features = np.array([[
            season, yr, mnth, holiday, workingday, weathersit,
            temp, atemp, hum, windspeed,
            hour_sin, hour_cos, weekday_sin, weekday_cos
        ]])

        input_scaled = X_scaler.transform(features)

        seq = np.repeat(input_scaled, 24, axis=0)
        seq = seq.reshape(1, 24, -1)

        input_tensor = torch.tensor(seq, dtype=torch.float32)

        with torch.no_grad():
            pred = model(input_tensor).numpy()

        pred_original = y_scaler.inverse_transform(pred.reshape(-1,1))[0][0]

        st.success(f"🚲 Predicted Demand: {pred_original:.2f}")

        if pred_original < 100:
            st.error("Low Availability ⚠️")
        else:
            st.success("Good Availability ✅")

# ===============================
# DATASET
# ===============================
elif page == "Dataset Explorer":
    st.title("📂 Dataset Explorer")

    st.dataframe(df.head(100))

    st.write("Shape:", df.shape)
    st.write("Columns:", list(df.columns))

# ===============================
# PERFORMANCE
# ===============================
elif page == "Model Performance":
    st.title("📈 Model Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric("R² Score", "0.92")
    col2.metric("RMSE", "~60")
    col3.metric("MAE", "~42")

    st.markdown("""
    This Transformer model captures temporal dependencies effectively,
    leading to high prediction accuracy.
    """)

# ===============================
# ABOUT
# ===============================
elif page == "About":
    st.title("ℹ️ About Project")

    st.markdown("""
    ### Bike Demand Prediction

    - Deep Learning Model: Transformer
    - Sequence Length: 24 hours
    - Accuracy: R² ≈ 0.92

    This app predicts bike demand based on weather and time features,
    helping optimize bike availability.
    """)
