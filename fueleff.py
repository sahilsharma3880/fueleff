import streamlit as st
import numpy as np
import joblib

model = joblib.load("abc11.joblib")     
scaler = joblib.load("scaler.pkl")   

st.title("🚗 MPG Prediction App")

cylinders = st.number_input("Cylinders", 4.0, 10.0, step=1.0)
displacement = st.number_input("Displacement", 50.0, 500.0, step=1.0)
horsepower = st.number_input("Horsepower", 40.0, 250.0, step=1.0)
weight = st.number_input("Weight", 1500.0, 6000.0, step=10.0)
acceleration = st.number_input("Acceleration", 8.0, 25.0, step=0.1)
model_year = st.number_input("Model Year", 50.0, 90.0, step=1.0)

origin = st.selectbox("Origin", [1, 2, 3])

if st.button("Predict MPG"):

    input_data = np.array([[cylinders,
                            displacement,
                            horsepower,
                            weight,
                            acceleration,
                            model_year,
                            origin]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    pred_value = prediction[0] 

    st.success(f"Predicted MPG: {pred_value:.2f}")