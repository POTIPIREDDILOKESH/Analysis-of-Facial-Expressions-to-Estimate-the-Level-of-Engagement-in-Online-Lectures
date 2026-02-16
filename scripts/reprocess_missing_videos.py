import os
import subprocess

# Folders
VIDEOS_FOLDER = "../videos"
OPENFACE_EXE = "../OpenFace/FeatureExtraction.exe"
OUTPUT_FOLDER = "../openface_output"

# Get all video files
all_videos = [os.path.join(root, f)
              for root, dirs, files in os.walk(VIDEOS_FOLDER)
              for f in files if f.endswith(".avi") or f.endswith(".mp4")]

# Get all processed CSVs
processed_csvs = [f.replace(".csv", "")
                  for f in os.listdir(OUTPUT_FOLDER) if f.endswith(".csv")]

# Detect missing videos
missing_videos = [v for v in all_videos if os.path.splitext(os.path.basename(v))[0] not in processed_csvs]

print(f"Missing videos ({len(missing_videos)}):")
for v in missing_videos:
    print("-", v)

# Re-run OpenFace for missing videos
for video in missing_videos:
    print(f"\nProcessing missing video: {video}")
    subprocess.run([OPENFACE_EXE, "-f", video, "-out_dir", OUTPUT_FOLDER])

print("\n✅ All missing videos processed!")
