import pandas as pd
import os
import glob

# Path to all OpenFace CSVs
csv_folder = r"C:\Users\shiva\Downloads\EngagementDetectionProject\openface_output"
output_file = r"C:\Users\shiva\Downloads\EngagementDetectionProject\dataset\master_dataset.csv"

# List all CSV files
all_files = glob.glob(os.path.join(csv_folder, "*.csv"))

# Create an empty list to store dataframes
df_list = []

for file in all_files:
    df = pd.read_csv(file)
    
    # Add participant_id and video_file from filename
    filename = os.path.basename(file)
    df['video_file'] = filename
    
    # Extract participant ID (assumes format P_XX or PXX in filename)
    if '_' in filename:
        df['participant_id'] = filename.split('_')[0]
    else:
        df['participant_id'] = filename[:3]
    
    df_list.append(df)

# Concatenate all dataframes
master_df = pd.concat(df_list, ignore_index=True)

# Save as master dataset
master_df.to_csv(output_file, index=False)
print(f"✅ Master dataset saved! Shape: {master_df.shape}")
