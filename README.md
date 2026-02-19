# 🎯 Real-Time Facial Expression & Engagement Detection System

A Deep Learning-based system that analyzes video input to:

* 🔍 Detect **facial expressions (frame-by-frame)**
* 📊 Predict **overall engagement level (LOW / MEDIUM / HIGH)**
* 🎥 Generate an output video with real-time overlays

This project combines **OpenFace**, **Random Forest**, and **BiLSTM neural networks** to create a full video-based engagement analysis pipeline.

---

## 🚀 Features

✅ Automatic OpenFace feature extraction
✅ Frame-level facial expression prediction
✅ Video-level engagement classification
✅ Real-time overlay on output video
✅ Handles new unseen videos
✅ Fully modular training & inference pipeline

---

## 🏗️ Architecture Overview

### 1️⃣ Feature Extraction

* Tool: **OpenFace**
* Extracts:

  * Head pose (p_rx, p_ry, p_rz)
  * Action Units (AU01_r, AU02_r, …)
  * Facial landmarks

### 2️⃣ Expression Model

* Algorithm: **Random Forest**
* Input: OpenFace AU + head pose features
* Output: Expression class (per frame)

### 3️⃣ Engagement Model

* Algorithm: **BiLSTM + LSTM**
* Input: Sequential OpenFace features (+ optional expression fusion)
* Output:

  * LOW
  * MEDIUM
  * HIGH

---

## 🧠 Model Architecture

### Expression Model

```
OpenFace Features → StandardScaler → RandomForest → Expression Label
```

### Engagement Model

```
Sequential Features (per video)
        ↓
Masking Layer
        ↓
Bidirectional LSTM (64 units)
        ↓
LSTM (32 units)
        ↓
Dense + Dropout
        ↓
Softmax → Engagement Level
```

---

## 📂 Project Structure

```
EngagementDetectionProject/
│
├── dataset/
│   ├── master_dataset.csv
│   └── labeled_dataset.csv
│
├── models/
│   ├── expression_model.pkl
│   ├── expression_scaler.pkl
│   ├── engagement_model_lstm_rnn.h5
│   └── engagement_scaler.pkl
│
├── scripts/
│   ├── train_expression_model.py
│   ├── train_model.py
│   └── predict_final_video.py
│
├── OpenFace/
│
└── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv engagement_env
engagement_env\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

Required libraries:

* tensorflow
* scikit-learn
* pandas
* numpy
* opencv-python
* joblib

---

## 🏋️ Training

### Train Expression Model

```bash
python train_expression_model.py
```

This will generate:

```
models/expression_model.pkl
models/expression_scaler.pkl
models/expression_label_encoder.pkl
```

---

### Train Engagement Model

```bash
python train_model.py
```

This will generate:

```
models/engagement_model_lstm_rnn.h5
models/engagement_scaler.pkl
models/engagement_label_encoder.pkl
```

---

## 🎥 Run Prediction on New Video

```bash
python predict_final_video.py --video "../input/test_video.mp4"
```

### Output:

* Extracts features using OpenFace
* Predicts frame-level expressions
* Predicts overall engagement
* Saves annotated video:

```
test_video_output.mp4
```

Overlay Example:

```
Expression: Angry
Engagement: LOW
```

---

## 📊 Engagement Levels

| Class | Meaning |
| ----- | ------- |
| 0     | LOW     |
| 1     | MEDIUM  |
| 2     | HIGH    |

---

## 🔬 Dataset Description

The dataset contains:

* Video ID
* Frame-level OpenFace features
* Expression labels
* Video-level engagement label

Expression is predicted per frame.
Engagement is predicted per video sequence.

---

## 💡 Future Improvements

* 🔄 Real-time webcam support
* 📈 Attention mechanism in LSTM
* 🌐 Web deployment (Streamlit / Flask)
* 📊 Temporal smoothing for engagement

---

## 🎓 Applications

* Online learning engagement monitoring
* Classroom attention analysis
* Behavioral research
* Human-computer interaction studies
* Interview performance analytics

---

## 👨‍💻 Author

Developed as part of a deep learning research project on video-based engagement estimation.

