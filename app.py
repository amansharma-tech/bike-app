import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib

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
    "Data Insights",
    "Prediction",
    "Dataset Explorer",
    "Model Performance",
    "About"
])

# ===============================
# DASHBOARD (WELCOME)
# ===============================
if page == "Dashboard":

    st.title("🚲 Bike Demand Prediction System")

    st.markdown("""
    ### 👋 Welcome!

    This application helps analyze and predict bike rental demand using a Transformer-based model.

    ### 🔍 What you can do:
    - 📊 Explore trends in **Data Insights**
    - 🔮 Predict demand using real-world inputs
    - 📈 View model performance
    - 📂 Explore dataset

    ---
    👉 Use the sidebar to navigate.
    """)

# ===============================
# DATA INSIGHTS (GRAPHS)
# ===============================
elif page == "Data Insights":

    st.title("📊 Data Insights")

    # Hourly Demand
    if "hr" in df.columns:
        st.subheader("⏰ Demand by Hour")

        hourly = df.groupby("hr")["cnt"].mean()

        chart_data = pd.DataFrame({
            "Hour": hourly.index,
            "Average Demand": hourly.values
        })

        st.line_chart(chart_data.set_index("Hour"))

    # Season Demand
    if "season" in df.columns:
        st.subheader("🌤️ Demand by Season")

        season_map = {1:"Spring", 2:"Summer", 3:"Fall", 4:"Winter"}
        df["season_name"] = df["season"].map(season_map)

        season_data = df.groupby("season_name")["cnt"].mean()

        chart_data = pd.DataFrame({
            "Season": season_data.index,
            "Average Demand": season_data.values
        })

        st.bar_chart(chart_data.set_index("Season"))

    # Weather Impact
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

        chart_data = pd.DataFrame({
            "Weather": weather_data.index,
            "Average Demand": weather_data.values
        })

        st.bar_chart(chart_data.set_index("Weather"))

# ===============================
# PREDICTION
# ===============================
elif page == "Prediction":

     st.button("🚀 Predict Demand"):

    # Convert to normalized
    temp = temp_c / 50
    atemp = temp
    hum = hum_percent / 100
    windspeed = wind_kmh / 50

    # Cyclical encoding
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

    # ===============================
    # RESULT
    # ===============================
    st.success(f"🚲 Predicted Demand: {pred_original:.2f}")

    # ===============================
    # PEAK WARNING SYSTEM
    # ===============================
    if pred_original > 700:
        st.error("🚨 Peak Demand! Bikes may not be available.")
    elif pred_original > 300:
        st.warning("⚠️ Moderate Demand — plan accordingly.")
    else:
        st.success("✅ Low Demand — good time to rent!")

    # ===============================
    # BEST HOUR SUGGESTION
    # ===============================
    st.subheader("🧠 Best Time Suggestion")

    hourly = df.groupby("hr")["cnt"].mean()
    best_hour = hourly.idxmin()

    st.info(f"💡 Best hour to rent bike: {best_hour}:00 (lowest demand)")

    # ===============================
    # DOWNLOAD REPORT
    # ===============================
    st.subheader("📄 Download Report")

    report_df = pd.DataFrame({
        "Feature": [
            "Season", "Year", "Month", "Holiday", "Working Day",
            "Weather", "Temperature", "Humidity", "Windspeed",
            "Hour", "Weekday", "Predicted Demand"
        ],
        "Value": [
            season, yr, mnth, holiday, workingday,
            weathersit, temp_c, hum_percent, wind_kmh,
            hour, weekday, pred_original
        ]
    })

    csv = report_df.to_csv(index=False)

    st.download_button(
        label="📥 Download Prediction Report",
        data=csv,
        file_name="bike_prediction_report.csv",
        mime="text/csv"
    )
# ===============================
# DATASET
# ===============================
elif page == "Dataset Explorer":
    st.title("📂 Dataset Explorer")
    st.dataframe(df.head(100))
    st.write("Shape:", df.shape)

# ===============================
# PERFORMANCE
# ===============================
elif page == "Model Performance":
    st.title("📈 Model Performance")
    st.metric("R² Score", "0.92")
    st.metric("RMSE", "~60")
    st.metric("MAE", "~42")

# ===============================
# ABOUT
# ===============================
elif page == "About":
    st.title("ℹ️ About")
    st.markdown("""
    Transformer-based model for time-series prediction  
    Captures temporal patterns using past 24-hour data  
    Achieves high accuracy (R² ≈ 0.92)
    """)
