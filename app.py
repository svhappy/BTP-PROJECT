import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="ECDM ML Dashboard",
    layout="wide"
)

st.title("⚡ ECDM Stability Prediction Dashboard")
st.write("Machine Learning Based Analysis of ECDM Voltage Signals")

# ---------------- LOAD MODEL ----------------
model = joblib.load("ecdm_stability_model.pkl")
scaler = joblib.load("ecdm_scaler.pkl")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload CSV Voltage Signal File",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        # ---------------- LOAD DATA ----------------
        df = pd.read_csv(uploaded_file, skiprows=2, header=None)
        df = df.apply(pd.to_numeric, errors='coerce').dropna()

        time = df.iloc[:, 0].values
        voltage = df.iloc[:, 1].values

        # ---------------- PEAK DETECTION ----------------
        peaks, _ = find_peaks(
            voltage,
            height=20,
            prominence=15,
            distance=10
        )

        # ---------------- FEATURE EXTRACTION ----------------
        max_voltage = np.max(voltage)
        mean_voltage = np.mean(voltage)
        std_voltage = np.std(voltage)
        discharge_count = len(peaks)

        frequency = (
            1 / np.mean(np.diff(time[peaks]))
            if len(peaks) > 1 else 0
        )

        duty_cycle = (
            100 * np.sum(voltage > (0.5 * max_voltage))
            / len(voltage)
        )

        if len(peaks) > 1:
            intervals = np.diff(time[peaks])
            max_gap = np.max(intervals)
            gap_std = np.std(intervals)
        else:
            max_gap = 0
            gap_std = 0

        # ---------------- FEATURE DATAFRAME ----------------
        features = pd.DataFrame([{
            "max_voltage": max_voltage,
            "mean_voltage": mean_voltage,
            "std_voltage": std_voltage,
            "discharge_count": discharge_count,
            "frequency": frequency,
            "duty_cycle": duty_cycle,
            "max_gap": max_gap,
            "gap_std": gap_std
        }])

        # ---------------- SCALE FEATURES ----------------
        features_scaled = scaler.transform(features)

        # ---------------- PREDICTION ----------------
        prediction = model.predict(features_scaled)[0]

        if prediction == 1:
            result = "✅ STABLE"
        else:
            result = "❌ UNSTABLE"

        # ---------------- DISPLAY RESULT ----------------
        st.subheader("Prediction Result")
        st.success(result)

        # ---------------- METRICS ----------------
        st.subheader("Extracted Signal Features")

        col1, col2, col3 = st.columns(3)

        col1.metric("Max Voltage", f"{max_voltage:.2f} V")
        col1.metric("Mean Voltage", f"{mean_voltage:.2f} V")

        col2.metric("Frequency", f"{frequency:.2f} Hz")
        col2.metric("Duty Cycle", f"{duty_cycle:.2f} %")

        col3.metric("Discharges", discharge_count)
        col3.metric("Max Gap", f"{max_gap:.6f}")

        # ---------------- PLOT ----------------
        st.subheader("Voltage-Time Signal")

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.plot(time, voltage, label="Voltage Signal")

        if len(peaks) > 0:
            ax.scatter(
                time[peaks],
                voltage[peaks],
                color='red',
                label='Discharges'
            )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Voltage (V)")
        ax.set_title("ECDM Voltage Signal")
        ax.grid(True)
        ax.legend()

        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
