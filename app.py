import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# Title
st.title("🎓 Student Performance Predictor")

st.markdown(
"""
This Machine Learning app predicts a student's **Exam Score**
based on study habits and background factors.
"""
)

st.write("### Enter Student Details")

# Hours studied
hours_studied = st.slider("Hours Studied", 0, 12, 4)

# Attendance
attendance = st.slider("Attendance (%)", 0, 100, 75)

# Parental involvement dropdown
parental_option = st.selectbox(
    "Parental Involvement",
    ["Low", "Medium", "High"]
)

parental_map = {"Low":0, "Medium":1, "High":2}
parental_involvement = parental_map[parental_option]

# Distance from home dropdown
distance_option = st.selectbox(
    "Distance From Home",
    ["Near", "Moderate", "Far"]
)

distance_map = {"Near":0, "Moderate":1, "Far":2}
distance_from_home = distance_map[distance_option]

# Gender dropdown
gender_option = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

gender = 1 if gender_option == "Male" else 0

# Predict button
if st.button("Predict Exam Score"):

    features = np.array([[hours_studied, attendance, parental_involvement, distance_from_home, gender]])

    prediction = model.predict(features)

    st.success(f"Predicted Exam Score: {prediction[0]:.2f}")