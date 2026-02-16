# train_model_finetune.py
import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Masking
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, accuracy_score
import lightgbm as lgb

# -----------------------------
# 1️⃣ Load labeled dataset
# -----------------------------
print("Loading labeled dataset...")
df = pd.read_csv("../dataset/labeled_dataset.csv")

# Drop unnecessary columns
feature_cols = [c for c in df.columns if c not in ['video_id', 'engagement']]
X = df[feature_cols].values
y = df['engagement'].values

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# -----------------------------
# 2️⃣ Prepare sequences by video
# -----------------------------
print("Preparing sequences by video...")
sequences = []
labels = []

for vid, group in df.groupby('video_id'):
    seq = scaler.transform(group[feature_cols].values)  # normalize per frame
    sequences.append(seq)
    labels.append(group['engagement'].iloc[0])  # label per video

# Pad sequences
max_len = max([len(seq) for seq in sequences])
X_seq = pad_sequences(sequences, maxlen=max_len, dtype='float32', padding='post')
y_seq = np.array(labels)

# Encode labels to integers
le = LabelEncoder()
y_seq = le.fit_transform(y_seq)

# -----------------------------
# 3️⃣ Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.25, random_state=42, stratify=y_seq
)
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# -----------------------------
# 4️⃣ Compute class weights for imbalance
# -----------------------------
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(weights))
print("Class weights:", class_weights)

# -----------------------------
# 5️⃣ Build Bidirectional LSTM model
# -----------------------------
print("Building Bidirectional LSTM model...")
model = Sequential()
model.add(Masking(mask_value=0., input_shape=(max_len, len(feature_cols))))
model.add(Bidirectional(LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)))
model.add(LSTM(32, dropout=0.3, recurrent_dropout=0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(len(np.unique(y_seq)), activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# -----------------------------
# 6️⃣ Train model with early stopping
# -----------------------------
es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=8,
    validation_split=0.1,
    class_weight=class_weights,
    callbacks=[es]
)

# -----------------------------
# 7️⃣ Evaluate model
# -----------------------------
print("Evaluating model...")
y_pred = np.argmax(model.predict(X_test), axis=1)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -----------------------------
# 8️⃣ Save the model and scaler
# -----------------------------
os.makedirs("../models", exist_ok=True)
model.save("../models/engagement_model_lstm_rnn.h5")
print("✅ Model saved to ../models/engagement_model_lstm_rnn.h5")

# Save the scaler
import pickle
with open("../models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("✅ Scaler saved to ../models/scaler.pkl")

# -----------------------------
# 9️⃣ Optional: LightGBM fallback
# -----------------------------
use_lightgbm = True
if use_lightgbm:
    print("\nTraining LightGBM as alternative...")
    # Aggregate features per video
    video_features = df.groupby('video_id')[feature_cols].mean()
    labels = df.groupby('video_id')['engagement'].first()
    X_train_lgb, X_test_lgb, y_train_lgb, y_test_lgb = train_test_split(
        video_features, labels, test_size=0.25, random_state=42, stratify=labels
    )

    clf = lgb.LGBMClassifier(class_weight='balanced')
    clf.fit(X_train_lgb, y_train_lgb)
    y_pred_lgb = clf.predict(X_test_lgb)
    print("LightGBM Accuracy:", accuracy_score(y_test_lgb, y_pred_lgb))
    print("\nLightGBM Classification Report:\n", classification_report(y_test_lgb, y_pred_lgb))
