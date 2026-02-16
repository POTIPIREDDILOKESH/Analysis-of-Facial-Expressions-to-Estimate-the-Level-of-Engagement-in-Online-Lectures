import os
import subprocess
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import argparse

def predict_engagement(video_path, openface_exe):
    """
    Input: video file path
    Output: predicted engagement level (Low, Medium, High)
    """

    # -----------------------------
    # Paths
    # -----------------------------
    output_dir = r"C:\Users\shiva\Downloads\EngagementDetectionProject\openface_output"
    model_path = r"C:\Users\shiva\Downloads\EngagementDetectionProject\models\engagement_model_lstm_rnn.h5"

    os.makedirs(output_dir, exist_ok=True)

    # -----------------------------
    # Run OpenFace on the video
    # -----------------------------
    print("Running OpenFace...")
    subprocess.run([
        openface_exe,
        "-f", video_path,
        "-out_dir", output_dir
    ], check=True)

    # -----------------------------
    # Load OpenFace CSV
    # -----------------------------
    csv_file = os.path.join(output_dir, os.path.basename(video_path).replace(".avi", ".csv"))
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"OpenFace output not found: {csv_file}")

    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip()

    # Remove non-feature columns if present
    feature_cols = [c for c in df.columns if c not in ['video_id', 'engagement']]

    X_seq = df[feature_cols].values.astype('float32')

    # -----------------------------
    # Fix feature mismatch with model
    # -----------------------------
    expected_features = 714  # your trained model expects 714 features per frame
    current_features = X_seq.shape[1]

    if current_features < expected_features:
        print(f"Warning: OpenFace returned {current_features} features, adding zeros to match {expected_features}")
        X_seq = np.hstack([X_seq, np.zeros((X_seq.shape[0], expected_features - current_features), dtype='float32')])
    elif current_features > expected_features:
        print(f"Warning: OpenFace returned {current_features} features, truncating to {expected_features}")
        X_seq = X_seq[:, :expected_features]

    # -----------------------------
    # Pad or truncate sequence to 300 frames
    # -----------------------------
    max_frames = 300
    X_seq_padded = pad_sequences([X_seq], maxlen=max_frames, dtype='float32', padding='post', truncating='post')[0]
    X_seq_padded = np.expand_dims(X_seq_padded, axis=0)  # add batch dimension

    # -----------------------------
    # Load model and predict
    # -----------------------------
    model = load_model(model_path)
    pred_prob = model.predict(X_seq_padded)
    pred_label = np.argmax(pred_prob, axis=1)[0]

    label_map = {0: "Low", 1: "Medium", 2: "High"}
    return label_map[pred_label]

# -----------------------------
# Command-line interface
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--openface", type=str, required=True, help="Path to OpenFace FeatureExtraction.exe")
    args = parser.parse_args()

    engagement = predict_engagement(args.video, args.openface)
    print(f"Predicted engagement for video: {engagement}")
