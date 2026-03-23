import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Bike Demand Predictor", layout="wide")

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv("hour.csv")

# ===============================
# LOAD SCALERS
# ===============================
X_scaler = joblib.load("x_scaler.pkl")
y_scaler = joblib.load("y_scaler.pkl")

# ===============================
# MODEL CLASSES
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
# LOAD MODEL
# ===============================
input_dim = X_scaler.n_features_in_
model = TransformerModel(input_dim)
model.load_state_dict(torch.load("transformer_model.pth", map_location="cpu"))
model.eval()

# ===============================
# SIDEBAR
# ===============================
st.sidebar.title("🚲 Bike Demand App")
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

    st.title("🚲 Bike Demand Analytics Dashboard")

    st.markdown("""
    Welcome to the **Bike Demand Prediction System** 🚀  
    This app helps analyze and predict bike rental demand based on time, weather, and seasonal factors.
    """)

    # KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rides", f"{int(df['cnt'].sum()):,}")
    col2.metric("Average Demand", f"{int(df['cnt'].mean())}")
    col3.metric("Peak Demand", f"{int(df['cnt'].max())}")

    st.markdown("---")

    # Hourly Graph
    if "hr" in df.columns:
        hourly = df.groupby("hr")["cnt"].mean()

        fig, ax = plt.subplots()
        ax.plot(hourly.index, hourly.values)
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Average Demand")
        ax.set_title("Hourly Bike Demand")

        st.pyplot(fig)
        st.info(f"Peak demand occurs around {hourly.idxmax()}:00")

    # Season Graph
    if "season" in df.columns:
        season_map = {1:"Spring", 2:"Summer", 3:"Fall", 4:"Winter"}
        df["season_name"] = df["season"].map(season_map)

        season_data = df.groupby("season_name")["cnt"].mean()

        fig, ax = plt.subplots()
        ax.bar(season_data.index, season_data.values)
        ax.set_xlabel("Season")
        ax.set_ylabel("Average Demand")
        ax.set_title("Demand by Season")

        st.pyplot(fig)

    # Weather Graph
    if "weathersit" in df.columns:
        weather_map = {
            1: "Clear",
            2: "Mist",
            3: "Light Rain",
            4: "Heavy Rain"
        }

        df["weather_name"] = df["weathersit"].map(weather_map)
        weather_data = df.groupby("weather_name")["cnt"].mean()

        fig, ax = plt.subplots()
        ax.bar(weather_data.index, weather_data.values)
        ax.set_xlabel("Weather Condition")
        ax.set_ylabel("Average Demand")
        ax.set_title("Weather Impact")

        st.pyplot(fig)

# ===============================
# PREDICTION
# ===============================
elif page == "Prediction":

    st.title("🔮 Predict Bike Demand")

    # Dictionaries
    season_dict = {"Spring":1, "Summer":2, "Fall":3, "Winter":4}
    weather_dict = {"Clear":1, "Mist":2, "Light Rain/Snow":3, "Heavy Rain":4}
    year_dict = {"2011":0, "2012":1}
    binary_dict = {"No":0, "Yes":1}

    col1, col2 = st.columns(2)

    with col1:
        season = season_dict[st.selectbox("Season", list(season_dict.keys()))]
        yr = year_dict[st.selectbox("Year", list(year_dict.keys()))]
        mnth = st.slider("Month", 1, 12, 6)
        holiday = binary_dict[st.selectbox("Holiday", list(binary_dict.keys()))]
        workingday = binary_dict[st.selectbox("Working Day", list(binary_dict.keys()))]
        weathersit = weather_dict[st.selectbox("Weather", list(weather_dict.keys()))]

    with col2:
        temp = st.slider("Temperature", 0.0, 1.0, 0.5)
        atemp = st.slider("Feels Like Temp", 0.0, 1.0, 0.5)
        hum = st.slider("Humidity", 0.0, 1.0, 0.5)
        windspeed = st.slider("Windspeed", 0.0, 1.0, 0.2)
        hour = st.slider("Hour", 0, 23, 12)
        weekday = st.slider("Weekday (0=Sun)", 0, 6, 3)

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
            st.error("Low Bike Availability ⚠️")
        else:
            st.success("Good Bike Availability ✅")

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

# ===============================
# ABOUT
# ===============================
elif page == "About":
    st.title("ℹ️ About Project")

    st.markdown("""
    This project uses a **Transformer-based deep learning model**
    to predict bike demand using time-series data.

    Key highlights:
    - Captures temporal patterns
    - Uses cyclical feature encoding
    - Achieves high accuracy (R² ≈ 0.92)
    """)
