import os
import pickle

import torch
from torch.utils.data import Dataset
import tensorflow.compat.v1 as tf

CAMERAS = ["FRONT", "FRONT_LEFT", "SIDE_LEFT", "FRONT_RIGHT", "SIDE_RIGHT"]


class WaymoDataset(Dataset):
    """Dataset for image segmentation and regression."""

    def __init__(
        self,
        scope="training",
        cameras=CAMERAS,
        order="random",
        exclusions=None,
        heatmaps=True,
    ):

        self.heatmaps = heatmaps

        # Create list of all filepaths
        self.filepaths = []
        root_path = os.getcwd() + f"/data/{scope}"
        for cam in cameras:
            cam_filepaths = os.listdir(f"{root_path}/{cam}")
            self.filepaths += [f"{root_path}/{cam}/{i}" for i in cam_filepaths]

        # Filter exclusions
        if exclusions is not None:
            self.filepaths = [
                i for i in self.filepaths if not any(j in i for j in exclusions)
            ]

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, item):

        sample = load_pickle(self.filepaths[item])
        img = torch.tensor(tf.image.decode_jpeg(sample["image"]).numpy())

        if self.heatmaps:
            labels = convert_to_heatmap(sample["labels"].labels)
        else:
            labels = sample["labels"].labels
        return {"img": img, "labels": labels}

    def get_context(self, item):
        return load_pickle(self.filepaths[item]["context"])


def load_pickle(fpath):
    with open(fpath, "rb") as f:
        return pickle.load(f)


def convert_to_heatmap(img_dims, labels):
    # TODO
    return
