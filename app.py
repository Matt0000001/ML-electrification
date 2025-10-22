import streamlit as st
import joblib
import tensorflow as tf
import numpy as np

# Load models
electrification_model = joblib.load("electrification_model.pkl")
project_cost_model = tf.keras.models.load_model("project_cost_model.keras", compile=False)

st.title("Electrification AI + Project Cost Estimator")

# Dropdown options
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

# One-hot encoding order
solution_labels = ["distribution line extension", "home system", "regular connection", "microgrid"]

# Inputs
location_classification = st.selectbox("Brgy - Island / Upland / Coastal", list(location_options.keys()))
household_grouping = st.selectbox("Scattered or Clustered", list(household_options.keys()))
mode_of_transportation = st.selectbox("Mode of Transportation", list(transportation_options.keys()))
distance = st.number_input("Distance (in km)", min_value=0.0)
target_households = st.number_input("Target Households", min_value=0)

if st.button("Get Suggestions"):
    # Prepare input for electrification model
    electrification_input = [
        location_options[location_classification],
        household_options[household_grouping],
        transportation_options[mode_of_transportation],
        distance,
        target_households
    ]

    # Predict probabilities for each solution
    electrification_solution_proba = electrification_model.predict_proba([electrification_input])[0]

    # Get top two predictions with their probabilities
    top_solution_indices = np.argsort(electrification_solution_proba)[::-1][:2]
    top_electrification_solutions = [
        (electrification_model.classes_[i], electrification_solution_proba[i]) for i in top_solution_indices
    ]

    # Extract labels and probabilities
    top_label = top_electrification_solutions[0][0]
    top_prob = top_electrification_solutions[0][1] * 100
    second_label = top_electrification_solutions[1][0]
    second_prob = top_electrification_solutions[1][1] * 100


    # One-hot encode both suggestions
    top_one_hot = [1 if top_label == label else 0 for label in electrification_model.classes_]
    second_one_hot = [1 if second_label == label else 0 for label in electrification_model.classes_]

    # Final input: one-hot + distance + target_households
    top_cost_input = np.array([top_one_hot + [distance, target_households]])
    second_cost_input = np.array([second_one_hot + [distance, target_households]])

    # Predict project cost for both suggestions
    top_project_cost = project_cost_model.predict(top_cost_input)[0][0]
    second_project_cost = project_cost_model.predict(second_cost_input)[0][0]

# Display top two suggestions with their estimated costs
st.success(
    f"Suggested Electrification Solution: {top_label} ({top_prob:.2f}%)\nEstimated Cost: ₱{top_project_cost * target_households:,.2f}"
)
st.info(
    f"Second Best Suggestion: {second_label} ({second_prob:.2f}%)\nEstimated Cost: ₱{second_project_cost * target_households:,.2f}"
)
