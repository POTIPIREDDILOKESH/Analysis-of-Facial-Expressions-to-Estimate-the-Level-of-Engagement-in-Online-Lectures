import os
import cv2
import argparse
import subprocess
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ==============================
# ARGUMENT
# ==============================
parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True, help="Input video path")
args = parser.parse_args()

video_path = args.video
video_name = os.path.splitext(os.path.basename(video_path))[0]
video_dir = os.path.dirname(video_path)
csv_path = os.path.join(video_dir, video_name + ".csv")

print("\n[INFO] Video:", video_path)

# ==============================
# RUN OPENFACE IF NEEDED
# ==============================
if not os.path.exists(csv_path):

    print("[INFO] Running OpenFace...")
    openface_exe = r"..\OpenFace\FeatureExtraction.exe"

    subprocess.run([
        openface_exe,
        "-f", video_path,
        "-out_dir", video_dir
    ])

    if not os.path.exists(csv_path):
        raise RuntimeError("❌ OpenFace failed.")

# ==============================
# LOAD CSV
# ==============================
print("[INFO] Loading OpenFace CSV...")
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

feature_cols = [
    'p_rx','p_ry','p_rz',
    'AU01_r','AU02_r','AU04_r','AU05_r',
    'AU06_r','AU07_r','AU09_r','AU10_r',
    'AU12_r','AU14_r','AU15_r','AU17_r',
    'AU23_r','AU26_r'
]

df = df[feature_cols].dropna()
print("Feature shape:", df.shape)

# ==============================
# LOAD MODELS
# ==============================
print("[INFO] Loading models...")

expression_model = joblib.load("../models/expression_model.pkl")
expression_scaler = joblib.load("../models/expression_scaler.pkl")
expression_encoder = joblib.load("../models/expression_encoder.pkl")

engagement_model = load_model("../models/engagement_model_lstm_rnn.h5")
engagement_scaler = joblib.load("../models/engagement_scaler.pkl")

# ==============================
# FRAME LEVEL EXPRESSION
# ==============================
print("[INFO] Predicting frame-level expressions...")

X_expr = expression_scaler.transform(df)
expr_pred = expression_model.predict(X_expr)

expr_labels = expression_encoder.inverse_transform(expr_pred)

# Encode expression again for engagement input
expr_encoded = expression_encoder.transform(expr_labels)

# Add expression column
df["expression"] = expr_encoded

# ==============================
# VIDEO LEVEL ENGAGEMENT
# ==============================
print("[INFO] Predicting video-level engagement...")

# IMPORTANT: scale INCLUDING expression column
eng_feature_cols = feature_cols + ["expression"]

df_scaled = df.copy()
df_scaled[eng_feature_cols] = engagement_scaler.transform(df_scaled[eng_feature_cols])

X_eng = np.expand_dims(df_scaled[eng_feature_cols].values, axis=0)

eng_probs = engagement_model.predict(X_eng, verbose=0)
eng_class = int(np.argmax(eng_probs))

label_map = {
    0: "LOW",
    1: "MEDIUM",
    2: "HIGH"
}

engagement_label = label_map.get(eng_class, "UNKNOWN")

print("Predicted Engagement:", engagement_label)

# ==============================
# VIDEO OUTPUT
# ==============================
print("\n[INFO] Creating output video...")

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 25

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_video = os.path.join(video_dir, video_name + "_output.mp4")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get frame expression safely
    if frame_idx < len(expr_labels):
        expr_text = expr_labels[frame_idx]
    else:
        expr_text = "Unknown"

    cv2.putText(frame,
                f"Expression: {expr_text}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2)

    cv2.putText(frame,
                f"Engagement: {engagement_label}",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    writer.write(frame)
    frame_idx += 1

cap.release()
writer.release()

print("✅ Output saved:", output_video)
