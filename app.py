import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from category_encoders.target_encoder import TargetEncoder
from sklearn.ensemble import RandomForestClassifier


st.title("🧠 Mental Health Depression Prediction App")
st.write("Upload dataset terlebih dahulu agar model dapat mempelajari pola data.")


# ==========================
# 📌 STEP 1: LOAD / UPLOAD DATASET
# ==========================
file_upload = st.file_uploader("Upload student_depression_dataset.csv", type=["csv"])

if file_upload:
    df = pd.read_csv(file_upload)
    st.success("📁 Dataset berhasil di-load!")
else:
    st.warning("⚠ Harap upload file dataset sebelum lanjut.")
    st.stop()


# ==========================
# 📌 STEP 2: CLEANING + ENCODING SETUP
# ==========================

# Drop column yang tidak dipakai
drop_cols = ['Work Pressure', 'Job Satisfaction']
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

# Mapping ordinal fixed values
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
    df_copy = df.copy()
    for col, mapping in ordinal_mapping.items():
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].astype(str).map(mapping).fillna(0)
    return df_copy


df = apply_ordinal(df)


# Label Encoding
label_cols = ['Gender', 'Dietary Habits', 'Degree']
label_encoders = {}

for col in label_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le


# Target Encoding
target_cols = ['City', 'Profession']
te = TargetEncoder()

if set(target_cols).issubset(df.columns):
    df[target_cols] = te.fit_transform(df[target_cols], df["Depression"])


# Scaling
scaler = StandardScaler()
feature_cols = [col for col in df.columns if col != "Depression"]
df_scaled = df.copy()
df_scaled[feature_cols] = scaler.fit_transform(df_scaled[feature_cols])


# ==========================
# 📌 STEP 3: Train Model Once
# ==========================

model = RandomForestClassifier(random_state=42)
model.fit(df_scaled[feature_cols], df_scaled["Depression"])


# ==========================
# 📌 STEP 4: UI INPUT FORM
# ==========================

st.subheader("🧾 Masukkan data untuk prediksi:")

input_data = {}

for col in feature_cols:
    if df[col].dtype == "object":
        input_data[col] = st.selectbox(col, sorted(df[col].unique()))
    elif col in ordinal_mapping:
        input_data[col] = st.selectbox(col, list(ordinal_mapping[col].keys()))
    else:
        input_data[col] = st.number_input(col, value=float(df[col].mean()))


# ==========================
# 📌 STEP 5: Convert User Input → Model Format
# ==========================

def preprocess_user_input(data_dict):

    data = pd.DataFrame([data_dict])

    # Apply ordinal encoding
    data = apply_ordinal(data)

    # Apply label encoding
    for col, encoder in label_encoders.items():
        if col in data.columns:
            data[col] = encoder.transform([str(data[col].values[0])])[0]

    # Apply target encoding
    for col in target_cols:
        if col in data.columns:
            data[col] = te.transform(pd.DataFrame({col: [data[col].values[0]]}))[col][0]

    # Apply scaling
    data[feature_cols] = scaler.transform(data[feature_cols])

    return data


# ==========================
# 📌 STEP 6: Predict
# ==========================

if st.button("🔍 Predict"):
    user_processed = preprocess_user_input(input_data.copy())
    pred = model.predict(user_processed)[0]

    if pred == 1:
        st.error("⚠️ Kamu menunjukkan indikasi depresi. Pertimbangkan dukungan profesional.")
    else:
        st.success("💚 Tidak ada indikasi depresi berdasarkan inputmu. Tetap jaga kesehatan mental ya!")


st.write("---")
st.caption("⚠️ Model ini hanya alat estimasi dan tidak menggantikan diagnosis medis.")
