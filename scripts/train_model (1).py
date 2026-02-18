# ===============================
# FINAL train_model.py
# Engagement BiLSTM Model
# ===============================

import os
import pandas as pd
import numpy as np
import pickle
import joblib

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Masking
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample
from sklearn.metrics import classification_report, accuracy_score

print("Loading dataset...")
df = pd.read_csv("../dataset/master_dataset.csv")

# -----------------------------------
# Safety check
# -----------------------------------
required_cols = ["video_id", "engagement", "expression"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"❌ Missing column: {col}")

# -----------------------------------
# Balance dataset
# -----------------------------------
balanced = []
max_count = df['engagement'].value_counts().max()

for label in df['engagement'].unique():
    balanced.append(
        resample(
            df[df['engagement'] == label],
            replace=True,
            n_samples=max_count,
            random_state=42
        )
    )

df = pd.concat(balanced)

# -----------------------------------
# Feature Columns (ONLY these)
# -----------------------------------
feature_cols = [
    'p_rx','p_ry','p_rz',
    'AU01_r','AU02_r','AU04_r','AU05_r',
    'AU06_r','AU07_r','AU09_r','AU10_r',
    'AU12_r','AU14_r','AU15_r','AU17_r',
    'AU23_r','AU26_r'
]

# -----------------------------------
# Encode Expression (important)
# -----------------------------------
expr_encoder = LabelEncoder()
df["expression"] = expr_encoder.fit_transform(df["expression"])
joblib.dump(expr_encoder, "../models/expression_encoder.pkl")

# Add expression as feature
feature_cols.append("expression")

# -----------------------------------
# Scale Features
# -----------------------------------
scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

# Save feature column order (VERY IMPORTANT)
joblib.dump(feature_cols, "../models/engagement_feature_cols.pkl")

# -----------------------------------
# Build Sequences
# -----------------------------------
sequences = []
labels = []

for vid, grp in df.groupby('video_id'):
    sequences.append(grp[feature_cols].values)
    labels.append(grp['engagement'].iloc[0])

max_len = max(len(s) for s in sequences)

X_seq = pad_sequences(
    sequences,
    maxlen=max_len,
    padding='post',
    dtype='float32'
)

# Encode engagement labels
le = LabelEncoder()
y_seq = le.fit_transform(labels)
joblib.dump(le, "../models/engagement_label_encoder.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq,
    test_size=0.25,
    random_state=42,
    stratify=y_seq
)

# -----------------------------------
# Model
# -----------------------------------
model = Sequential([
    Masking(mask_value=0., input_shape=(max_len, len(feature_cols))),
    Bidirectional(LSTM(64, return_sequences=True, dropout=0.3)),
    LSTM(32, dropout=0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(len(np.unique(y_seq)), activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=8,
    validation_split=0.1,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
)

y_pred = np.argmax(model.predict(X_test), axis=1)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# -----------------------------------
# Save Everything
# -----------------------------------
os.makedirs("../models", exist_ok=True)

model.save("../models/engagement_model_lstm_rnn.h5")

with open("../models/engagement_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Engagement model saved successfully")
