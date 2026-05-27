import streamlit as st
import pickle
import numpy as np

# loading saved model
model = pickle.load(
    open("models/dt_model.pkl", "rb")
)

st.set_page_config(
    page_title="Heart Disease Prediction",
    layout="centered"
)

st.title("Heart Disease Prediction using Decision Tree")

st.write("Enter patient details below")

age = st.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=45
)

sex = st.selectbox(
    "Sex",
    ["Male", "Female"]
)

cp = st.slider(
    "Chest Pain Type",
    0,
    3,
    1
)

trestbps = st.number_input(
    "Resting Blood Pressure",
    value=120
)

chol = st.number_input(
    "Cholesterol",
    value=200
)

thalach = st.number_input(
    "Maximum Heart Rate",
    value=150
)

oldpeak = st.number_input(
    "Oldpeak",
    value=1.0
)

sex = 1 if sex == "Male" else 0

# creating input array
input_data = np.array([[
    age,
    sex,
    cp,
    trestbps,
    chol,
    0,
    0,
    0,
    0,
    thalach,
    0,
    oldpeak,
    1
]])

if st.button("Predict"):

    prediction = model.predict(input_data)

    if prediction[0] == 1:

        st.error(
            "High Chance of Heart Disease"
        )

    else:

        st.success(
            "Low Chance of Heart Disease"
        )