import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from category_encoders.target_encoder import TargetEncoder
import joblib

# Load dataset
df = pd.read_csv("student_depression_dataset.csv")

# ==================== CLEANING ======================
df['Sleep Duration'] = df['Sleep Duration'].astype(str).str.replace("'", "").str.strip()

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
    df[col] = df[col].map(mapping).fillna(0).astype(float)

# ==================== LABEL ENCODING BERES ====================
label_cols = ['Gender', 'Dietary Habits', 'Degree']
label_encoders = {}

for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# ==================== TARGET ENCODING ====================
target_enc_cols = ['City', 'Profession']
te = TargetEncoder()
df[target_enc_cols] = te.fit_transform(df[target_enc_cols], df['Depression'])

# Drop unused columns
df = df.drop(columns=['Work Pressure', 'Job Satisfaction'])

# Remove nulls & duplicates
df = df.dropna().drop_duplicates()

# ==================== SCALING ====================
scaler = StandardScaler()
feature_columns = [col for col in df.columns if col != "Depression"]
df[feature_columns] = scaler.fit_transform(df[feature_columns])

# Split Dataset
X = df.drop(columns=['Depression'])
y = df['Depression']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==================== TRAIN MODEL ====================
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ==================== SAVE MODEL & ENCODERS ====================
joblib.dump(model, "depression_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(te, "target_encoder.pkl")

print("\n🎉 Model Saved Successfully!")
