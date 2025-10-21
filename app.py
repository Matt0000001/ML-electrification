import streamlit as st
import joblib

# Load your trained model
model = joblib.load("electrification_model.pkl")

st.title("Electrification AI Model")

# Define dropdown options based on your mappings
location_options = {
    "Rolling": 1,
    "Flatland": 2,
    "Coastal": 3,
    "Upland": 4,
    "Island": 5
}

household_options = {
    "Clustered": 1,
    "Scattered": 2
}

transportation_options = {
    "None": 0,
    "vehicle": 2,
    "motorcycle/Habal habal": 3,
    "Boat": 4,
    "Horse/Carabao/Cow": 5,
    "Motorcycle/HabalHabal": 6
}

# Dropdowns for categorical features
location_classification = st.selectbox("Brgy - Island / Upland / Coastal", list(location_options.keys()))
household_grouping = st.selectbox("Scattered or Clustered", list(household_options.keys()))
mode_of_transportation = st.selectbox("Mode of Transportation", list(transportation_options.keys()))

# Numeric inputs
distance = st.number_input("Distance (in km)", min_value=0.0)
target_households = st.number_input("Target Households", min_value=0)

if st.button("Suugestion"):
    input_data = [[
        location_options[location_classification],
        household_options[household_grouping],
        transportation_options[mode_of_transportation],
        distance,
        target_households
    ]]
    prediction = model.predict(input_data)
    st.success(f"Suggested Electrification Solution: {prediction[0]}")
