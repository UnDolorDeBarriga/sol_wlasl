import os
import shutil
import json

# Load the JSON data
with open('start_kit/WLASL_parsed_data_adjustedpath.json', 'r') as f:
    data = json.load(f)

# Base directory where the new folders will be created
base_dir = 'data'
moved_files = 0
missing_files = 0

for item in data:
    # Get the current path to the video
    current_path = item['video_path']

    # Check if the file exists before taking the next steps
    if os.path.exists(current_path):
    
        # Get the split (train/test/val) and gloss (label) from the JSON item
        split = item['split']
        gloss = item['gloss']

        # Create the split and gloss directories if they don't exist
        split_dir = os.path.join(base_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        gloss_dir = os.path.join(split_dir, gloss)
        os.makedirs(gloss_dir, exist_ok=True)

        # Create the new path of the video
        new_path = os.path.join(gloss_dir, os.path.basename(current_path))

        # Move the video to the new directory
        shutil.copy(current_path, new_path)

        moved_files += 1

        # print(f"the video {current_path} is moved to {new_path}")
    else:
        missing_files += 1
        print(f"the video {current_path} does not exist")
    
print(f"Moved {moved_files} files and {missing_files} files are missing")