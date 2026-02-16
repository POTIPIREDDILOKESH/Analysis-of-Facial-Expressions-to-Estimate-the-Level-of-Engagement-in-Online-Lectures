🎓 Analysis of Facial Expressions to Estimate the Level of Engagement in Online Lectures




📌 1. Project Objective

Online education lacks real-time engagement feedback.
This project automatically estimates student engagement levels from facial expressions recorded in lecture videos.

The system:

Detects faces

Extracts facial action units

Models temporal behavior using LSTM

Predicts engagement level

📊 2. Dataset
DAiSEE Dataset

Dataset link:
https://people.iith.ac.in/vineethnb/resources/daisee/index.html

The DAiSEE dataset contains:

Student lecture videos

Annotated engagement levels

Real classroom scenarios

Multiple participants

Multiple sessions per participant

📁 3. Project Structure
EngagementDetectionProject/
│
├── OpenFace/                     # OpenFace toolkit
├── videos/                       # Raw videos (P_01, P_02, etc.)
├── openface_output/              # CSV files from OpenFace
├── dataset/
│   ├── master_dataset.csv
│   ├── labels.csv
│   └── labeled_dataset.csv
│
├── models/
│   ├── engagement_model_lstm_rnn.h5
│   └── scaler.pkl
│
├── scripts/
│   ├── run_openface_all.bat
│   ├── combine_csv.py
│   ├── generate_labels.py
│   ├── add_labels.py
│   ├── train_model.py
│   └── predict_from_video.py
│
└── README.md

🔁 4. Complete System Workflow
Lecture Video
    ↓
Face Detection (OpenFace)
    ↓
Facial Landmark Tracking
    ↓
Action Unit Extraction
Head Pose + Eye Gaze
    ↓
Frame-level CSV Data
    ↓
Confidence Filtering
    ↓
Sequence Preparation
    ↓
Bidirectional LSTM + RNN
    ↓
Engagement Classification

🧠 5. OpenFace Feature Extraction

OpenFace extracts:

68 facial landmarks

Head pose (pose_Rx, pose_Ry, pose_Rz)

Eye gaze vectors

Action Units (AU01–AU45)

Detection confidence

Command used:

FeatureExtraction.exe -f video.avi -out_dir openface_output


Batch processing:

run_openface_all.bat


Each video generates a CSV file containing ~700+ features per frame.

🧹 6. Data Preprocessing

After extraction, preprocessing is performed:

Step 1 — Remove Low Confidence Frames
confidence > 0.8
success == 1


This ensures:

No false detections

No noisy facial readings

Step 2 — Combine All CSV Files
python combine_csv.py


Output:

master_dataset.csv


Contains:

All videos combined

Video ID column

Cleaned frame-level features

Step 3 — Label Generation

Engagement score is calculated using:

Engagement Score = (Mean(AU12_c) + (1 - Mean(AU45_c))) / 2


Where:

AU12 → Smile

AU45 → Blink

Labels assigned:

0 → Low

1 → Medium

2 → High

Command:

python generate_labels.py

Step 4 — Merge Labels
python add_labels.py


Creates:

labeled_dataset.csv

📦 7. Sequence Preparation for LSTM

LSTM requires time-series input.

For each video:

Frames are grouped

Converted into sequences

Padded to uniform length (max_len = 300)

Final shape:

(samples, 300, 714)

🤖 8. Deep Learning Architecture

Model structure:

Masking Layer

Bidirectional LSTM (64 units)

LSTM (32 units)

Dense Layer (32 units)

Dropout (0.3)

Softmax Output Layer

Why Bidirectional?

Captures forward and backward temporal dependencies

Better understanding of engagement evolution

📈 9. Model Training

Command:

python train_model.py


Training details:

Loss: sparse_categorical_crossentropy

Optimizer: Adam

Epochs: 50

Batch Size: 8

Validation Split: 0.1

Class Weights applied

📊 10. Model Performance

Deep Learning Model Accuracy: ~85%
Alternative LightGBM Model: ~94%

Performance Metrics:

Precision

Recall

F1-score

Confusion Matrix

🎥 11. Real-Time Video Prediction

To test with new video:

python predict_from_video.py --video "input/test1.avi" --openface "OpenFace/FeatureExtraction.exe"


Pipeline:

OpenFace extracts features

Features are cleaned

Sequence is padded

Model predicts engagement level

Output:

Predicted Engagement: Low / Medium / High

🔬 12. Feature Engineering Details

Key features used:

Facial Action Units

Head pose variation

Eye blink frequency

Smile intensity

Temporal behavioral patterns

Temporal modeling improves accuracy compared to static frame classification.

🚀 13. Installation Guide
Create Virtual Environment
python -m venv engagement_env_tf
engagement_env_tf\Scripts\activate

Install Dependencies
pip install tensorflow pandas numpy scikit-learn matplotlib

🔧 14. OpenFace Installation

Download OpenFace

Extract into project root

Test:

OpenFace\FeatureExtraction.exe -h

💡 15. Future Enhancements

Transformer-based architecture

Real-time webcam dashboard

Attention heatmaps

Web deployment (Flask / Streamlit)

Larger labeled dataset

📘 16. Research Contribution

This project demonstrates:

Practical facial behavior modeling

Temporal engagement tracking

Deep learning applied to education analytics

Real-world emotion analytics pipeline

👨‍💻 _______Lokesh P.

