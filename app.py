import streamlit as st
import tensorflow as tf
import numpy as np

# Load updated electrification model
electrification_model = tf.keras.models.load_model("tfelectrification_model_v3.h5", compile=False)

# Load individual cost models (no scaling)
project_cost_models = {
    "distribution line extension": tf.keras.models.load_model("project_cost_DLE_no_scale_model.keras", compile=False),
    "home system": tf.keras.models.load_model("project_cost_home_syste_no_scale_model.keras", compile=False),
    "regular connection": tf.keras.models.load_model("project_cost_regular_connection_no_scale_model.keras", compile=False)
}

# Streamlit UI
st.title("Electrification AI + Project Cost Estimator")

# Dropdown options
location_options = ["Rolling", "Flatland", "Coastal", "Upland", "Island"]
household_options = ["Clustered", "Scattered"]
transportation_options = ["None", "vehicle", "motorcycle/Habal habal", "Boat", "Horse/Carabao/Cow"]

# One-hot encoding order
solution_labels = ["distribution line extension", "home system", "regular connection", "microgrid"]

# Inputs
location_classification = st.selectbox("Brgy - Island / Upland / Coastal", location_options)
household_grouping = st.selectbox("Scattered or Clustered", household_options)
mode_of_transportation = st.selectbox("Mode of Transportation", transportation_options)
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

# Prepare one-hot encoded input
def prepare_input(location, household, transportation, distance, target_households):
    location_one_hot = [
        1 if location == "Rolling" else 0,
        1 if location == "Flatland" else 0,
        1 if location == "Coastal" else 0,
        1 if location == "Upland" else 0,
        1 if location == "Island" else 0
    ]
    household_one_hot = [
        1 if household == "Clustered" else 0,
        1 if household == "Scattered" else 0
    ]
    transportation_clean = transportation.lower().replace(" ", "")
    transportation_one_hot = [
        1 if transportation == "None" else 0,
        1 if transportation == "vehicle" else 0,
        1 if transportation == "Boat" else 0,
        1 if transportation == "Horse/Carabao/Cow" else 0,
        1 if "motorcycle/habalhabal" in transportation_clean else 0
    ]
    return location_one_hot + household_one_hot + transportation_one_hot + [distance, target_households]

if st.button("Get Suggestions"):
    electrification_input = prepare_input(location_classification, household_grouping, mode_of_transportation, distance, target_households)
    electrification_input = np.array([electrification_input])

    # Predict probabilities for each solution
    electrification_solution_proba = electrification_model.predict(electrification_input)[0]

    # Get top two predictions with their probabilities
    top_solution_indices = np.argsort(electrification_solution_proba)[::-1][:2]
    top_electrification_solutions = [(solution_labels[i], electrification_solution_proba[i]) for i in top_solution_indices]

    # Extract labels and probabilities
    top_label, top_prob = top_electrification_solutions[0]
    second_label, second_prob = top_electrification_solutions[1]

    # Compute costs
    top_project_cost = compute_cost(top_label, distance, target_households)
    second_project_cost = compute_cost(second_label, distance, target_households)

    # Display top two suggestions with their estimated costs
    st.success(
        f"Suggested Electrification Solution: {top_label} ({top_prob * 100:.2f}%)\n"
        f"Estimated Cost: ₱{top_project_cost:,.2f}"
    )
    st.info(
        f"Second Best Suggestion: {second_label} ({second_prob * 100:.2f}%)\n"
        f"Estimated Cost: ₱{second_project_cost:,.2f}"
    )
