import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from category_encoders.target_encoder import TargetEncoder
from sklearn.ensemble import RandomForestClassifier


# ==========================
# 🔧 Load Dataset for Encoding Reference
# ==========================
df = pd.read_csv("student_depression_dataset.csv")   # harus ada di folder yang sama


# ==========================
# 🔧 PREPROCESSING FUNCTION
# ==========================

def preprocess_input(data):

    df_copy = df.copy()

    # -------- Drop Columns --------
    drop_cols = ['Work Pressure', 'Job Satisfaction']
    df_copy = df_copy.drop(columns=drop_cols)

    # -------- Encoding Rules --------
    ordinal_mapping = {
        "Sleep Duration": {
            "Less than 5 hours": 1,
            "5-6 hours": 2,
            "7-8 hours": 3,
            "More than 8 hours": 4,
            "Others": 0
        },
        "Financial Stress": {
            "1.0": 1, "2.0": 2, "3.0": 3, "4.0": 4, "5.0": 5, "?": 0
        },
        "Have you ever had suicidal thoughts ?": {"No": 0, "Yes": 1},
        "Family History of Mental Illness": {"No": 0, "Yes": 1}
    }

    for col, mapping in ordinal_mapping.items():
        data[col] = mapping.get(data[col], 0)

    # Label Encoding
    label_cols = ['Gender', 'Dietary Habits', 'Degree']

    for col in label_cols:
        le = LabelEncoder()
        df_copy[col] = df_copy[col].astype(str)
        le.fit(df_copy[col])
        data[col] = le.transform([str(data[col])])[0]

    # Target Encoding
    target_cols = ['City', 'Profession']
    te = TargetEncoder()
    df_copy[target_cols] = te.fit_transform(df_copy[target_cols], df_copy["Depression"])

    for col in target_cols:
        data[col] = te.transform(pd.DataFrame({col: [data[col]]}))[col][0]

    # Convert to DF
    df_input = pd.DataFrame([data])

    # Scaling
    scaler = StandardScaler()
    feature_cols = df_input.columns
    df_input[feature_cols] = scaler.fit_transform(df_input[feature_cols])

    return df_input



# ==========================
# 🚀 Load Model
# ==========================
model = RandomForestClassifier()
model.fit(
    preprocess_input(df.drop(columns=["Depression"])),
    df["Depression"]
)


# ==========================
# 🧠 STREAMLIT UI
# ==========================

st.title("🩺 Mental Health Depression Prediction App")
st.write("Masukkan data kamu lalu klik predict untuk melihat hasilnya.")


# Input Form
gender = st.selectbox("Gender", df["Gender"].unique())
city = st.selectbox("City", df["City"].unique())
profession = st.selectbox("Profession", df["Profession"].unique())
age = st.number_input("Age", min_value=10, max_value=80, step=1)
cgpa = st.number_input("CGPA", min_value=0.0, max_value=4.0, step=0.1)
hours = st.number_input("Work/Study Hours", min_value=0, max_value=20, step=1)
sleep = st.selectbox("Sleep Duration", df["Sleep Duration"].unique())
financial = st.selectbox("Financial Stress (1-5)", ["1.0", "2.0", "3.0", "4.0", "5.0"])
diet = st.selectbox("Dietary Habits", df["Dietary Habits"].unique())
degree = st.selectbox("Degree", df["Degree"].unique())
social = st.selectbox("Social Weakness", df["Social Weakness"].unique())
suicide = st.selectbox("Have suicidal thoughts?", ["Yes", "No"])
history = st.selectbox("Family History of Mental Illness", ["Yes", "No"])
academic = st.selectbox("Academic Pressure", df["Academic Pressure"].unique())
satisfaction = st.selectbox("Study Satisfaction", df["Study Satisfaction"].unique())


if st.button("🔍 Predict"):

    input_data = {
        "Gender": gender,
        "City": city,
        "Profession": profession,
        "Age": age,
        "CGPA": cgpa,
        "Work/Study Hours": hours,
        "Sleep Duration": sleep,
        "Dietary Habits": diet,
        "Degree": degree,
        "Social Weakness": social,
        "Have you ever had suicidal thoughts ?": suicide,
        "Financial Stress": financial,
        "Family History of Mental Illness": history,
        "Academic Pressure": academic,
        "Study Satisfaction": satisfaction
    }

    processed = preprocess_input(input_data)
    prediction = model.predict(processed)[0]

    if prediction == 1:
        st.error("⚠️ Kamu menunjukkan indikasi depresi. Sebaiknya konsultasi dengan profesional.")
    else:
        st.success("💚 Kamu tidak menunjukkan tanda depresi. Tetap jaga kesehatan mental ya!")
