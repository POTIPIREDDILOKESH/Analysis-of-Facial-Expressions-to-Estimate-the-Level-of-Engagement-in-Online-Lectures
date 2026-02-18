import pandas as pd
import numpy as np

print("Loading master dataset...")

df = pd.read_csv("../dataset/master_dataset.csv")
df.columns = df.columns.str.strip()

# =============================
# Use same AU + pose features
# =============================
feature_cols = [
    'p_rx','p_ry','p_rz',
    'AU01_r','AU02_r','AU04_r','AU05_r',
    'AU06_r','AU07_r','AU09_r','AU10_r',
    'AU12_r','AU14_r','AU15_r','AU17_r',
    'AU23_r','AU26_r'
]

df = df.dropna(subset=feature_cols)

print("Generating expression labels...")

# =============================
# Simple Rule-Based Expression Logic
# (You can later improve this)
# =============================
def detect_expression(row):
    
    # Smile
    if row['AU12_r'] > 1.5:
        return "happy"
    
    # Angry
    if row['AU04_r'] > 1.5 and row['AU07_r'] > 1.5:
        return "angry"
    
    # Surprise
    if row['AU05_r'] > 1.5 and row['AU26_r'] > 1.5:
        return "surprise"
    
    # Default
    return "neutral"


df["expression"] = df.apply(detect_expression, axis=1)

print(df["expression"].value_counts())

# Save updated dataset
df.to_csv("../dataset/master_dataset.csv", index=False)

print("✅ Expression column added successfully!")
