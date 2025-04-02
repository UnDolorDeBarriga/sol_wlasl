TF_ENABLE_ONEDNN_OPTS = 0

import warnings
warnings.filterwarnings("ignore", message=".*Failed to load image Python extension.*")
warnings.filterwarnings("ignore", message=".*torchvision.transforms.functional_tensor module is deprecated.*")
warnings.filterwarnings("ignore", message=".*weights of VideoMAEForVideoClassification were not initialized.*")
warnings.filterwarnings("ignore", message=".*The torchvision.datapoints and torchvision.*")

import os
import json
import pytorchvideo
import transformers 
import evaluate
import pathlib
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import pytorchvideo.data
import torch
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    RandomRotation,
    Resize,
    ElasticTransform,
)

def return_bigger_ds(n_gloss):
    dataset_root_path = pathlib.Path("data/train/")
    folder_video_counts = {}
    for folder in dataset_root_path.glob("*"):
        if folder.is_dir():
            video_count = len(list(folder.glob("*")))
            folder_video_counts[folder.name] = video_count
    sorted_folders = sorted(folder_video_counts.items(), key=lambda x: x[1], reverse=True)
    return [folder for folder, _ in sorted_folders[:n_gloss]]

class AddDistortion(torch.nn.Module):
    """
    Adds distortion to a video.
    """
    def __init__(self, distortion=0.5):
        super().__init__()
        self.distortion = distortion
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == 4, "video must have shape (C, T, H, W)"       
        # Create a new tensor with the same shape as x, filled with random values
        random_values = torch.rand_like(x) * 0 + np.random.normal(0, self.distortion)
        # Add the random values to x
        x = x + random_values
        return x

dataset_root_path = pathlib.Path("data")

# # all_video_file_paths = (
#     list(dataset_root_path.glob("train/*/*.mp4"))
#     + list(dataset_root_path.glob("val/*/*.mp4"))
#     + list(dataset_root_path.glob("test/*/*.mp4"))
# )
# print(all_video_file_paths[:5])

class_labels = [path.name for path in dataset_root_path.glob("test/*") if path.is_dir()]

set_class_labels = return_bigger_ds(10)

label2id = {label: i for i, label in enumerate(set_class_labels)}
id2label = {i: label for label, i in label2id.items()}

print(f"Label to id: {label2id}")

# print(len(class_labels))
# print(f"Unique classes: {list(label2id.keys())}.\nNumber of classes: {len(class_labels)}")


model_ckpt = "MCG-NJU/videomae-base"
image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
model = VideoMAEForVideoClassification.from_pretrained(
    model_ckpt,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
)

print("-------")

mean = image_processor.image_mean
std = image_processor.image_std
if "shortest_edge" in image_processor.size:
    height = width = image_processor.size["shortest_edge"]
else:
    height = image_processor.size["height"]
    width = image_processor.size["width"]
resize_to = (height, width)

num_frames_to_sample = model.config.num_frames
sample_rate = 4
fps = 30
clip_duration = num_frames_to_sample * sample_rate / fps



train_transform = Compose(
    [
        ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    # same arguments as test set
                    UniformTemporalSubsample(num_frames_to_sample),
                    Lambda(lambda x: x / 255.0),
                    Normalize(mean, std),
                    Resize(resize_to, antialias=True),
                    
                    # additional noise to avoid overfitting
                    RandomHorizontalFlip(p=0.4),
                    RandomRotation(degrees=10),
                    ElasticTransform(alpha=30.0),
                    AddDistortion(0.1),

                    # # Use generalized RandomTransformCustom for both RandomInvert and RandomAutocontrast
                    # RandomTransformCustom(RandomAutocontrast(p=1.0), p=0.2),  
                    # RandomTransformCustom(RandomInvert(p=1.0), p=0.3),        
                ]
            ),
        ),
    ]
)



train_dataset = pytorchvideo.data.Ucf101(
    data_path=os.path.join(dataset_root_path, "train"),
    clip_sampler=pytorchvideo.data.make_clip_sampler("random", clip_duration),
    decode_audio=False,
    transform=train_transform,
)


val_transform = Compose(
    [
        ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames_to_sample),
                    Lambda(lambda x: x / 255.0),
                    Normalize(mean, std),
                    Resize(resize_to),
                ]
            ),
        ),
    ]
)

val_dataset = pytorchvideo.data.Ucf101(
    data_path=os.path.join(dataset_root_path, "val"),
    clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
    decode_audio=False,
    transform=val_transform,
)

test_dataset = pytorchvideo.data.Ucf101(
    data_path=os.path.join(dataset_root_path, "test"),
    clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
    decode_audio=False,
    transform=val_transform,
)

# import imageio
# import numpy as np
# from IPython.display import Image

# def unnormalize_img(img):
#     """Un-normalizes the image pixels."""
#     img = (img * std) + mean
#     img = (img * 255).astype("uint8")
#     return img.clip(0, 255)

# def create_gif(video_tensor, filename="sample.gif"):
#     """Prepares a GIF from a video tensor.
#     The video tensor is expected to have the following shape:
#     (num_frames, num_channels, height, width).
#     """
#     frames = []
#     for video_frame in video_tensor:
#         frame_unnormalized = unnormalize_img(video_frame.permute(1, 2, 0).numpy())
#         frames.append(frame_unnormalized)
#     kargs = {"duration": 0.25}
#     imageio.mimsave(filename, frames, "GIF", **kargs)
#     return filename

# def display_gif(video_tensor, gif_name="sample.gif"):
#     """Prepares and displays a GIF from a video tensor."""
#     video_tensor = video_tensor.permute(1, 0, 2, 3)
#     gif_filename = create_gif(video_tensor, gif_name)
#     return Image(filename=gif_filename)

# sample_video = next(iter(train_dataset))
# video_tensor = sample_video["video"]
# display_gif(video_tensor)


