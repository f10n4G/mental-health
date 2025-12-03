import gdown
import os
import joblib

MODEL_PATH = "depression_model.pkl"

if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/file/d/1xQj2oRkAcQVZXYcUP1LFwHaYlOyViykZ/view?usp=sharing"
    gdown.download(url, MODEL_PATH, quiet=False)

model = joblib.load(MODEL_PATH)

# Load model & encoders
model = joblib.load("depression_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")
target_enc = joblib.load("target_encoder.pkl")

def predict_depression(input_dict):

    df = pd.DataFrame([input_dict])

    # Ordinal Mapping
    ordinal_mapping = {
        "Sleep Duration": {
            "Less than 5 hours": 1,
            "5-6 hours": 2,
            "7-8 hours": 3,
            "More than 8 hours": 4,
            "Others": 0
        },
        "Financial Stress": {"1.0": 1, "2.0": 2, "3.0": 3, "4.0": 4, "5.0": 5, "?": 0},
        "Have you ever had suicidal thoughts ?": {"No": 0, "Yes": 1},
        "Family History of Mental Illness": {"No": 0, "Yes": 1}
    }

    for col, mapping in ordinal_mapping.items():
        df[col] = df[col].map(mapping).fillna(0)

    # Apply Label Encoders
    for col in label_encoders:
        df[col] = label_encoders[col].transform(df[col].astype(str))

    # Apply Target Encoding
    target_cols = ["City", "Profession"]
    df[target_cols] = target_enc.transform(df[target_cols])

    # Scale features
    scaled = scaler.transform(df)

    # Predict
    prediction = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1]

    result = "Depression Detected" if prediction == 1 else "No Depression"
    
    return result, round(float(prob) * 100, 2)

