import streamlit as st
from prediction import predict_depression

st.title("🧠 Depression Risk Prediction System")
st.write("Fill in the form below and click Predict.")

inputs = {}

inputs["Age"] = st.number_input("Age", 10, 60, 20)
inputs["CGPA"] = st.number_input("CGPA", 0.0, 4.0, 3.0)
inputs["Work/Study Hours"] = st.number_input("Work/Study Hours", 0, 16, 6)

inputs["Gender"] = st.selectbox("Gender", ["Male", "Female", "Other"])
inputs["City"] = st.text_input("City", "Jakarta")
inputs["Profession"] = st.text_input("Profession", "Student")
inputs["Sleep Duration"] = st.selectbox(
    "Sleep Duration", 
    ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"]
)
inputs["Dietary Habits"] = st.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy"])
inputs["Degree"] = st.selectbox("Degree", ["Bachelor", "Highschool", "Master"])

inputs["Social Weakness"] = st.slider("Social Weakness", 0, 5, 3)
inputs["Have you ever had suicidal thoughts ?"] = st.selectbox(
    "Suicidal Thoughts", 
    ["Yes", "No"]
)
inputs["Financial Stress"] = st.selectbox(
    "Financial Stress Level", 
    ["1.0","2.0","3.0","4.0","5.0"]
)
inputs["Family History of Mental Illness"] = st.selectbox(
    "Family Mental Illness", 
    ["Yes", "No"]
)

inputs["Academic Pressure"] = st.slider("Academic Pressure", 1, 5, 3)
inputs["Study Satisfaction"] = st.slider("Study Satisfaction", 1, 5, 3)

if st.button("Predict"):
    result, prob = predict_depression(inputs)
    st.subheader(f"🔍 Result: **{result}**")
    st.write(f"📊 Estimated Risk Score: **{prob}%**")
