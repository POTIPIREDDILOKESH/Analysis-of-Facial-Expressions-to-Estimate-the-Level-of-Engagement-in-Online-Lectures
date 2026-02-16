import pandas as pd

print("Loading master dataset...")
# Load the full combined master dataset
master_path = "../dataset/master_dataset.csv"
df = pd.read_csv(master_path)

print("Loading labels...")
labels_path = "../dataset/labels.csv"
labels = pd.read_csv(labels_path)

# Ensure column names are correct
if list(labels.columns) != ["video_id", "engagement"]:
    labels.columns = ["video_id", "engagement"]

print("Merging labels with dataset...")
# Merge on video_id
df = df.merge(labels, on="video_id")

print("Saving labeled dataset...")
# Save the full labeled dataset
output_path = "../dataset/labeled_dataset.csv"
df.to_csv(output_path, index=False)

print("✅ Labeled dataset saved successfully!")
print("Final shape:", df.shape)
