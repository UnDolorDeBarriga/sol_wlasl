import os
import json
import pathlib


def remove_not_complete_ds():
    dataset_root_path = pathlib.Path("data")
    for folder in dataset_root_path.glob("test/*"):
        if folder.is_dir():
            if not (dataset_root_path / "val" / folder.name).exists() or not (dataset_root_path / "train" / folder.name).exists():
                print(f"Removing folder: {folder.name}")
                for sub_file in folder.glob("*"):
                    sub_file.unlink()  # Remove all files in the folder
                folder.rmdir()  # Remove the folder itself
    for folder in dataset_root_path.glob("val/*"):
        if folder.is_dir():
            if not (dataset_root_path / "test" / folder.name).exists() or not (dataset_root_path / "train" / folder.name).exists():
                print(f"Removing folder: {folder.name}")
                for sub_file in folder.glob("*"):
                    sub_file.unlink()  # Remove all files in the folder
                folder.rmdir()  # Remove the folder itself
    for folder in dataset_root_path.glob("train/*"):
        if folder.is_dir():
            if not (dataset_root_path / "test" / folder.name).exists() or not (dataset_root_path / "val" / folder.name).exists():
                print(f"Removing folder: {folder.name}")
                for sub_file in folder.glob("*"):
                    sub_file.unlink()  # Remove all files in the folder
                folder.rmdir()  # Remove the folder itself

def get_number_clusetr():
    dataset_root_path = pathlib.Path("data")
    test_label = [path.name for path in dataset_root_path.glob("test/*") if path.is_dir()]
    train_label = [path.name for path in dataset_root_path.glob("train/*") if path.is_dir()]
    val_label = [path.name for path in dataset_root_path.glob("val/*") if path.is_dir()]

    print(f"Len test lable: {len(test_label)}")
    print(f"Len train lable: {len(train_label)}")
    print(f"Len val lable: {len(val_label)}")

def return_bigger_ds(n_gloss):
    dataset_root_path = pathlib.Path("data/train/")
    folder_video_counts = {}
    for folder in dataset_root_path.glob("*"):
        if folder.is_dir():
            video_count = len(list(folder.glob("*")))
            folder_video_counts[folder.name] = video_count
    sorted_folders = sorted(folder_video_counts.items(), key=lambda x: x[1], reverse=True)
    return [folder for folder, _ in sorted_folders[:n_gloss]]

def dataset_formating(buigger_ds):
    dataset_root_path_1 = pathlib.Path("data")
    dataset_root_path_2 = pathlib.Path("new_dataset")
    
    for folder in dataset_root_path_1.glob("train/*"):
        if folder.is_dir():
            if folder.name in buigger_ds:
                fp_train = "train/" + folder.name
                fp_test = "test/" + folder.name
                fp_val = "val/" + folder.name

                new_folder_path = dataset_root_path_2 / fp_train
                new_folder_path.mkdir(parents=True, exist_ok=True)
                new_folder_path = dataset_root_path_2 / fp_test
                new_folder_path.mkdir(parents=True, exist_ok=True)
                new_folder_path = dataset_root_path_2 / fp_val
                new_folder_path.mkdir(parents=True, exist_ok=True)
                
                for sub_file_test in dataset_root_path_1.glob(f"test/{folder.name}/*"):
                    new_file_path = dataset_root_path_2 / "test" / folder.name / sub_file_test.name
                    new_file_path.write_bytes(sub_file_test.read_bytes())
                for sub_file_train in dataset_root_path_1.glob(f"train/{folder.name}/*"):
                    new_file_path = dataset_root_path_2 / "train" / folder.name / sub_file_train.name
                    new_file_path.write_bytes(sub_file_train.read_bytes())
                for sub_file_val in dataset_root_path_1.glob(f"val/{folder.name}/*"):
                    new_file_path = dataset_root_path_2 / "val" / folder.name / sub_file_val.name
                    new_file_path.write_bytes(sub_file_val.read_bytes())





if __name__ == "__main__":
    # remove_not_complete_ds()
    # get_number_clusetr()
    a = return_bigger_ds(10)
    dataset_formating(a)
