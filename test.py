import os
import json
data_path = 'data'

i = 0


print(len([file for file in os.listdir('start_kit/videos') if file.endswith('.mp4')]))


with open('start_kit/WLASL_parsed_data_adjustedpath.json', 'r') as f:
    data = json.load(f)

length = len(data)
print(length)  # Check the size of the data

import pathlib
dataset_root_path = "data"
dataset_root_path = pathlib.Path(dataset_root_path)

video_count_train = len(list(dataset_root_path.glob("train/*/*.mp4")))
video_count_val = len(list(dataset_root_path.glob("val/*/*.mp4")))
video_count_test = len(list(dataset_root_path.glob("test/*/*.mp4")))
video_total = video_count_train + video_count_val + video_count_test
print(f"Total videos: {video_total}")

all_video_file_paths = (
    list(dataset_root_path.glob("train/*/*.mp4"))
    + list(dataset_root_path.glob("val/*/*.mp4"))
    + list(dataset_root_path.glob("test/*/*.mp4"))
 )
all_video_file_paths[:5]