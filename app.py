import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Page configuration
st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓")

st.title("🎓 Student Performance Predictor")

st.markdown("""
This Machine Learning app predicts a student's **Exam Score**
based on study habits and background factors.
""")

# Load dataset
data = pd.read_csv("student_data.csv")

# Convert categorical columns
data['Parental_Involvement'] = data['Parental_Involvement'].astype('category').cat.codes
data['Distance_from_Home'] = data['Distance_from_Home'].astype('category').cat.codes
data['Gender'] = data['Gender'].astype('category').cat.codes

# Features and target
X = data[['Hours_Studied','Attendance','Parental_Involvement','Distance_from_Home','Gender']]
y = data['Exam_Score']

# Train model
model = RandomForestRegressor()
model.fit(X, y)

st.write("### Enter Student Details")

# Inputs
hours_studied = st.slider("Hours Studied", 0, 12, 4)
attendance = st.slider("Attendance (%)", 0, 100, 75)

parental_option = st.selectbox(
    "Parental Involvement",
    ["Low", "Medium", "High"]
)

parental_map = {"Low":0, "Medium":1, "High":2}
parental_involvement = parental_map[parental_option]

distance_option = st.selectbox(
    "Distance From Home",
    ["Near", "Moderate", "Far"]
)

distance_map = {"Near":0, "Moderate":1, "Far":2}
distance_from_home = distance_map[distance_option]

gender_option = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

gender = 1 if gender_option == "Male" else 0

# Prediction
if st.button("Predict Exam Score"):

    features = np.array([[hours_studied, attendance, parental_involvement, distance_from_home, gender]])

    prediction = model.predict(features)

    st.success(f"Predicted Exam Score: {prediction[0]:.2f}")
