# ===============================
# train_expression_model.py
# Expression RandomForest Model
# ===============================

import os
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import resample

print("Loading dataset...")

# Load master dataset (NOW must contain expression column)
df = pd.read_csv("../dataset/master_dataset.csv")

# ---------------------------------
# CHECK EXPRESSION COLUMN
# ---------------------------------
if "expression" not in df.columns:
    raise ValueError(
        "❌ 'expression' column not found in master_dataset.csv.\n"
        "Add expression labels before training."
    )

# ---------------------------------
# SAME 17 FEATURES USED EVERYWHERE
# ---------------------------------
feature_cols = [
    'p_rx','p_ry','p_rz',
    'AU01_r','AU02_r','AU04_r','AU05_r',
    'AU06_r','AU07_r','AU09_r','AU10_r',
    'AU12_r','AU14_r','AU15_r','AU17_r',
    'AU23_r','AU26_r'
]

# Keep only needed columns
df = df[feature_cols + ["expression"]].dropna()

print("\nExpression distribution BEFORE balancing:")
print(df["expression"].value_counts())

# ---------------------------------
# BALANCE DATASET
# ---------------------------------
balanced = []
max_count = df["expression"].value_counts().max()

for label in df["expression"].unique():
    balanced.append(
        resample(
            df[df["expression"] == label],
            replace=True,
            n_samples=max_count,
            random_state=42
        )
    )

df = pd.concat(balanced)

print("\nExpression distribution AFTER balancing:")
print(df["expression"].value_counts())

# ---------------------------------
# ENCODE LABELS
# ---------------------------------
le = LabelEncoder()
y = le.fit_transform(df["expression"])

# ---------------------------------
# SCALE FEATURES
# ---------------------------------
X = df[feature_cols]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------
# TRAIN TEST SPLIT
# ---------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------------------------
# TRAIN RANDOM FOREST
# ---------------------------------
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ---------------------------------
# EVALUATION
# ---------------------------------
print("\nModel Performance:")
print(classification_report(y_test, model.predict(X_test)))

# ---------------------------------
# SAVE MODEL FILES
# ---------------------------------
os.makedirs("../models", exist_ok=True)

joblib.dump(model, "../models/expression_model.pkl")
joblib.dump(scaler, "../models/expression_scaler.pkl")
joblib.dump(le, "../models/expression_label_encoder.pkl")

print("\n✅ Expression model saved successfully!")
