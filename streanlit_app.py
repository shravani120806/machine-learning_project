import streamlit as st

st.title("Test Streamlit App")
st.write("🎉 Your deployment is working!")

import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("model.pkl")

st.title("🚉 EV Fuel Station Type Predictor")
st.write("Enter station details to predict the Fuel Type Code (CNG, LNG, etc.)")

# User Inputs (change options if needed)
station_name = st.text_input("Station Name", "Spire Montgomery Operations Center")
state = st.text_input("State", "Alabama")
latitude = st.number_input("Latitude", value=32.3617)
longitude = st.number_input("Longitude", value=-86.2791)

# Create input DataFrame (match column order used during training)
input_df = pd.DataFrame({
    "Station Name": [station_name],
    "State": [state],
    "Latitude": [latitude],
    "Longitude": [longitude]
})

# Prediction
if st.button("Predict Fuel Type Code"):
    prediction = model.predict(input_df)
    st.success(f"Predicted Fuel Type Code: {int(prediction[0])}")

 
