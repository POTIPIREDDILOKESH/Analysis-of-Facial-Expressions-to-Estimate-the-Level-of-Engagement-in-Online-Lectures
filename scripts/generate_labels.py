import os
import pandas as pd

input_folder = "../openface_output"
output_file = "../dataset/labels.csv"

files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]
labels = []

for file in files:
    df = pd.read_csv(os.path.join(input_folder, file))
    
    # Strip spaces from column names
    df.columns = df.columns.str.strip()

    # Compute engagement using AU12_c (smile) and AU45_c (eye)
    smile_score = df['AU12_c'].mean()
    eye_score = 1 - df['AU45_c'].mean()

    engagement_score = (smile_score + eye_score) / 2

    # Assign label: 0=low, 1=medium, 2=high
    if engagement_score > 0.6:
        label = 2
    elif engagement_score > 0.3:
        label = 1
    else:
        label = 0

    labels.append({"video_id": file.replace(".csv", ""), "engagement": label})

labels_df = pd.DataFrame(labels)
labels_df.to_csv(output_file, index=False)
print(f"✅ Labels generated for {len(labels)} videos")
