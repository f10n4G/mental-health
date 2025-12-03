import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from category_encoders.target_encoder import TargetEncoder
from sklearn.ensemble import RandomForestClassifier


st.title("🧠 Mental Health Depression Prediction App")
st.write("Upload dataset terlebih dahulu agar model dapat dilatih.")


# ==========================
# 📌 Upload Dataset
# ==========================

file = st.file_uploader("Upload student_depression_dataset.csv", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.success("📁 Dataset berhasil dimuat!")
else:
    st.warning("⚠ Harap upload dataset dulu.")
    st.stop()


# ==========================
# 📌 Preprocessing Setup
# ==========================

drop_cols = ['Work Pressure', 'Job Satisfaction']
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

ordinal_mapping = {
    "Sleep Duration": {
        "Less than 5 hours": 1,
        "5-6 hours": 2,
        "7-8 hours": 3,
        "More than 8 hours": 4,
        "Others": 0
    },
    "Financial Stress": {str(i): i for i in range(1, 6)},
    "Have you ever had suicidal thoughts ?": {"No": 0, "Yes": 1},
    "Family History of Mental Illness": {"No": 0, "Yes": 1}
}


def apply_ordinal(df):
    df = df.copy()
    for col, mapping in ordinal_mapping.items():
        if col in df.columns:
            df[col] = df[col].astype(str).map(mapping).fillna(0)
    return df


df = apply_ordinal(df)


label_cols = ['Gender', 'Dietary Habits', 'Degree']
label_encoders = {}
mode_values = {}

for col in label_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        mode_values[col] = df[col].mode()[0]


target_cols = ['City', 'Profession']
te = TargetEncoder()
df[target_cols] = te.fit_transform(df[target_cols], df["Depression"])


scaler = StandardScaler()
feature_cols = [col for col in df.columns if col != "Depression"]
df_scaled = df.copy()
df_scaled[feature_cols] = scaler.fit_transform(df_scaled[feature_cols])


model = RandomForestClassifier(random_state=42)
model.fit(df_scaled[feature_cols], df_scaled["Depression"])


# ==========================
# 📌 Form Input Users
# ==========================

st.subheader("Masukkan data untuk prediksi:")

user_input = {}

for col in feature_cols:
    if col in ordinal_mapping:
        user_input[col] = st.selectbox(col, list(ordinal_mapping[col].keys()))
    elif col in label_cols:
        user_input[col] = st.selectbox(col, label_encoders[col].classes_)
    elif col in target_cols:
        user_input[col] = st.selectbox(col, sorted(df[col].unique()))
    else:
        user_input[col] = st.number_input(col, value=float(df[col].mean()))


# ==========================
# 📌 Preprocess User Input
# ==========================

def preprocess_input(data):
    data = pd.DataFrame([data])

    data = apply_ordinal(data)

    for col, encoder in label_encoders.items():
        if col in data.columns:
            val = str(data[col].values[0])
            data[col] = encoder.transform([val])[0] if val in encoder.classes_ else mode_values[col]

    # 🔥 FIX: Transform BOTH target columns at once
    if set(target_cols).issubset(data.columns):
        data[target_cols] = te.transform(data[target_cols])

    data[feature_cols] = scaler.transform(data[feature_cols])

    return data


# ==========================
# 📌 Predict
# ==========================

if st.button("🔍 Predict"):
    processed = preprocess_input(user_input.copy())
    pred = model.predict(processed)[0]

    if pred == 1:
        st.error("⚠️ Hasil menunjukkan indikasi depresi. Disarankan konsultasi profesional.")
    else:
        st.success("💚 Tidak ada indikasi depresi. Tetap jaga kesehatan mental!")


st.write("---")
st.caption("Aplikasi ini tidak menggantikan diagnosis medis profesional.")
