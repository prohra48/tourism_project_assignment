
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# =========================
# LOAD MODEL FROM HF
# =========================
MODEL_REPO = "prohra48/tourism-model"

model_path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename="model.joblib"
)

model = joblib.load(model_path)

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Tourism Predictor", layout="centered")

st.title("🌍 Wellness Tourism Package Predictor")
st.markdown("### Built with MLOps Pipeline 🚀")

st.write("Fill customer details to predict purchase behavior")

# =========================
# SIDEBAR INPUTS
# =========================
st.sidebar.header("Customer Details")

Age = st.sidebar.slider("Age", 18, 70, 30)
TypeofContact = st.sidebar.selectbox("Type of Contact", [0, 1])
CityTier = st.sidebar.selectbox("City Tier", [1, 2, 3])
Occupation = st.sidebar.selectbox("Occupation", [0, 1, 2])
Gender = st.sidebar.selectbox("Gender", [0, 1])
NumberOfPersonVisiting = st.sidebar.slider("Number of Persons Visiting", 1, 10, 2)
PreferredPropertyStar = st.sidebar.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
MaritalStatus = st.sidebar.selectbox("Marital Status", [0, 1, 2])
NumberOfTrips = st.sidebar.slider("Number of Trips", 0, 10, 2)
Passport = st.sidebar.selectbox("Passport", [0, 1])
OwnCar = st.sidebar.selectbox("Own Car", [0, 1])
NumberOfChildrenVisiting = st.sidebar.slider("Children Visiting", 0, 5, 0)
Designation = st.sidebar.selectbox("Designation", [0, 1, 2, 3, 4])
MonthlyIncome = st.sidebar.number_input("Monthly Income", min_value=1000)
PitchSatisfactionScore = st.sidebar.slider("Pitch Satisfaction Score", 1, 5, 3)
ProductPitched = st.sidebar.selectbox("Product Pitched", [0, 1, 2, 3, 4])
NumberOfFollowups = st.sidebar.slider("Number of Followups", 0, 10, 2)
DurationOfPitch = st.sidebar.slider("Duration of Pitch", 5, 60, 20)

# =========================
# CREATE INPUT DATAFRAME
# =========================
input_data = pd.DataFrame([{
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": Passport,
    "OwnCar": OwnCar,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "ProductPitched": ProductPitched,
    "NumberOfFollowups": NumberOfFollowups,
    "DurationOfPitch": DurationOfPitch
}])

# =========================
# PREDICTION
# =========================
if st.button("Predict"):

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(f"✅ Customer WILL BUY (Confidence: {probability:.2f})")
    else:
        st.error(f"❌ Customer WILL NOT BUY (Confidence: {probability:.2f})")

    st.write("### Input Summary")
    st.dataframe(input_data)
