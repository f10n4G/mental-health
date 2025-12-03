import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from category_encoders.target_encoder import TargetEncoder
from sklearn.ensemble import RandomForestClassifier

st.title("🧠 Mental Health Depression Prediction App")
st.write("Silakan upload dataset (CSV) terlebih dahulu jika belum otomatis terbaca.")


# ======================================================
# 📌 STEP 1: LOAD DATASET (Auto / Manual Upload)
# ======================================================

csv_file = "student_depression_dataset.csv"

try:
    df = pd.read_csv(csv_file)
    st.success("📁 Dataset ditemukan otomatis!")
except:
    st.warning("⚠ Dataset tidak ditemukan. Upload file CSV di bawah ini.")
    file_upload = st.file_uploader("Upload student_depression_dataset.csv", type=["csv"])
    
    if file_upload:
        df = pd.read_csv(file_upload)
        st.success("📁 Dataset berhasil diupload!")
    else:
        st.stop()  # berhenti sampai file ada


# ======================================================
# 📌 PREPROCESSING FUNCTION
# ======================================================

def preprocess(data, df_ref):

    df_copy = df_ref.copy()

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
    numeric_features = df_input.columns
    df_input[numeric_features] = scaler.fit_transform(df_input[numeric_features])

    return df_input


# ======================================================
# 📌 TRAIN MODEL (Menggunakan dataset yg ada)
# ======================================================

model = RandomForestClassifier()
model.fit(
    preprocess(df.drop(columns=["Depression"]).iloc[0].to_dict(), df),
    df["Depression"]
)


# ======================================================
# 📌 STREAMLIT INPUT FORM UI
# ======================================================

st.subheader("🧾 Masukkan Data Anda")

form = {}

for col in df.columns:
    if col == "Depression":
        continue
    elif df[col].dtype == "object":
        form[col] = st.selectbox(col, sorted(df[col].dropna().unique()))
    else:
        form[col] = st.number_input(col, value=float(df[col].mean()))


# ======================================================
# 📌 PREDICT BUTTON
# ======================================================

if st.button("🔍 Predict"):

    processed = preprocess(form.copy(), df)
    result = model.predict(processed)[0]

    if result == 1:
        st.error("⚠️ Kamu menunjukkan indikasi depresi. Pertimbangkan bantuan profesional.")
    else:
        st.success("💚 Kamu tidak menunjukkan tanda depresi. Tetap jaga kesehatan mental!")


st.write("---")
st.caption("Model ini bukan diagnosis medis. Untuk kondisi serius, hubungi profesional kesehatan mental.")
