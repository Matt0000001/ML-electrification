import streamlit as st
import joblib

# Load your trained model
model = joblib.load("electrification_model.pkl")

st.title("Electrification Model Demo")

# Numeric input fields for all features
location_classification = st.number_input("Location Classification (numeric)")
household_grouping = st.number_input("Household Grouping (numeric)")
Mode_of_transportation = st.number_input("Mode of Transportation (numeric)")
distance = st.number_input("Distance (in km)", min_value=0.0)
target_households = st.number_input("Target Households", min_value=0)

if st.button("Predict"):
    input_data = [[
        location_classification,
        household_grouping,
        Mode_of_transportation,
        distance,
        target_households
    ]]
    prediction = model.predict(input_data)
    st.success(f"Predicted Electrification Solution: {prediction[0]}")