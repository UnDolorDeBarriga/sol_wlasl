import os
import shutil
import json

# Load the JSON data from the read-only file
with open('WLASL_v0.3.json', 'r') as f:
    d_read = json.load(f)

# Open the writable JSON file
with open('WLASL_parsed_data_adjustedpath.json', 'r+') as f2:
    try:
        d_write = json.load(f2)
    except json.JSONDecodeError:
        # Initialize with an empty list if the file is empty or invalid
        d_write = []

    # Iterate through all items in the read-only JSON
    for item in d_read:
        # print(item['gloss'])
        for inst in item['instances']:
            video_path = inst['video_id'] + ".mp4"
            if video_path in os.listdir('videos'):
                n_entry = {
                    'gloss': item['gloss'],
                    'video_path': "start_kit/videos/" + video_path,
                    'frame_start': inst['frame_start'],
                    'frame_end': inst['frame_end'],
                    'split': inst['split']
                }

            # Append the new entry only if it doesn't already exist
                if n_entry not in d_write:
                    d_write.append(n_entry)

    # Write the updated data back to the writable JSON file
    f2.seek(0)
    json.dump(d_write, f2, indent=4)
    f2.truncate()