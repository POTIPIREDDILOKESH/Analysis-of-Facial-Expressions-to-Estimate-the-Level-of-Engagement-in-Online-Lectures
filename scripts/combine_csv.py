import pandas as pd
import os

input_folder = "../openface_output"
all_data = []

for file in os.listdir(input_folder):

    if file.endswith(".csv"):

        file_path = os.path.join(input_folder, file)
        print("Reading:", file)

        df = pd.read_csv(file_path)

        # ✅ REMOVE EXTRA SPACES FROM COLUMN NAMES
        df.columns = df.columns.str.strip()

        # ✅ Keep only successful detections
        df = df[(df["confidence"] > 0.8) & (df["success"] == 1)]

        # Add video id
        df["video_id"] = file.replace(".csv", "")

        all_data.append(df)

# Combine all videos
final_df = pd.concat(all_data, ignore_index=True)

print("Final dataset shape:", final_df.shape)

# Save dataset
os.makedirs("../dataset", exist_ok=True)
final_df.to_csv("../dataset/master_dataset.csv", index=False)

print("✅ Dataset saved successfully!")
