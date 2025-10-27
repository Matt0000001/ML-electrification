import streamlit as st
import joblib
import tensorflow as tf
import numpy as np

# Load electrification model
electrification_model = joblib.load("electrification_model.pkl")

# Load individual cost models (no scaling)
project_cost_models = {
    "distribution line extension": tf.keras.models.load_model("project_cost_DLE_no_scale_model.keras", compile=False),
    "home system": tf.keras.models.load_model("project_cost_home_syste_no_scale_model.keras", compile=False),
    "regular connection": tf.keras.models.load_model("project_cost_regular_connection_no_scale_model.keras", compile=False)
}

# Streamlit UI
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

# Cost computation function
def compute_cost(label, distance, target_households):
    if label == "microgrid":
        return 0
    model = project_cost_models[label]
    one_hot = [1 if label == sol else 0 for sol in solution_labels]
    input_features = np.array([one_hot + [distance, target_households]])
    cost = model.predict(input_features)[0][0]
    return cost

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

    # Compute costs
    top_project_cost = compute_cost(top_label, distance, target_households)
    second_project_cost = compute_cost(second_label, distance, target_households)

    # Display top two suggestions with their estimated costs
    st.success(
        f"Suggested Electrification Solution: {top_label} ({top_prob:.2f}%)\n"
        f"Estimated Cost: ₱{top_project_cost:,.2f}"
    )
    st.info(
        f"Second Best Suggestion: {second_label} ({second_prob:.2f}%)\n"
        f"Estimated Cost: ₱{second_project_cost:,.2f}"
    )
