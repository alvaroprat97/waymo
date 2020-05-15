import os
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import tensorflow.compat.v1 as tf
import cv2
import numpy as np
from pycocotools.coco import COCO


CAMERAS = ["FRONT", "FRONT_LEFT", "SIDE_LEFT", "FRONT_RIGHT", "SIDE_RIGHT"]
BIG_X = 1280
SMALL_X = 886
Y_DIM = 1920


# class WaymoDataset(Dataset):
#     """Dataset for image segmentation and regression."""

#     def __init__(
#         self,
#         network = "eff_det",
#         scope="training",
#         cameras=CAMERAS,
#         order="random",
#         exclusions=None,
#         heatmaps=True,
#     ):
#         self.network = network
#         self.heatmaps = heatmaps

#         # Create list of all filepaths
#         self.filepaths = []
#         root_path = os.getcwd() + f"/data/{scope}"
#         for cam in CAMERAS:
#             cam_filepaths = os.listdir(f"{root_path}/{cam}")
#             self.filepaths += [f"{root_path}/{cam}/{i}" for i in cam_filepaths]

#         # Filter exclusions
#         if exclusions is not None:
#             self.filepaths = [
#                 i for i in self.filepaths if not any(j in i for j in exclusions)
#             ]

#     def __len__(self):
#         return len(self.filepaths)

#     def __getitem__(self, item):
#         fpath = self.filepaths[item]
#         sample = load_pickle(fpath)
#         img = torch.tensor(tf.image.decode_jpeg(sample["image"]).numpy())
#         cam_type = fpath.split("/")[5]  # FRONT, SIDE_LEFT etc
        
#         labels = scale_labels(sample["labels"].labels, cam_type)
        
#         if self.network == "eff_det":
#             if self.heatmaps:
#                 labels = convert_to_heatmap(labels)

#                 return {"img": img, "labels": labels}
          
#         else:
#             target = {}
#             boxes = []
#             classes = []
#             for i in range(1,5):
#                 for obj in labels[i]:
#                     boxes.append([int(obj['x']-0.5*obj['width']), int(obj['y']-0.5*obj['length']), \
#                                   int(obj['x']+0.5*obj['width']), int(obj['y']+0.5*obj['length'])])
#                     classes.append(i)
#             boxes = torch.as_tensor(boxes, dtype=torch.int64)
#             classes = torch.as_tensor(classes, dtype=torch.float32)
            
#             img = img.resize_(2, 2, 3)
#             target['boxes'] = boxes
#             target['classes'] = classes
#             return img.permute(2, 0, 1), target
            
            

#     def get_context(self, item):
#         return self.img_names[item]

class WaymoDataset(Dataset):
    """Dataset for image segmentation and regression."""

    def __init__(
        self,
        scope="training",
        root_dir="/home/project_x/data/",
        cameras=CAMERAS,
        exclusions=None,
        transform=None
    ):

        self.transform = transform

        # Create list of all filepaths
        self.filepaths = []
        root_path = root_dir + scope
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

        fpath = self.filepaths[item]
        raw_sample = load_pickle(fpath)
        img = self.load_image(raw_sample['image'])
        annot = self.load_annotations(raw_sample["labels"].labels)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)
        sample = {'raw':raw_sample, 'id':fpath.split('/')[-1], **sample}
        return sample

    def get_cam_type(self, item):
        return self.filepaths[item].split('/')[5]

    def load_image(self, byte_img):
        img = tf.image.decode_jpeg(byte_img).numpy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # if len(img.shape) == 2:
        #     img = skimage.color.gray2rgb(img)

        return img.astype(np.float32) / 255.
    def load_annotations(self, annot):
        # get ground truth annotations
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations
        if len(annot) == 0:
            return annotations

        # parse annotations
        for idx, a in enumerate(annot):

            # some annotations have basically no width / height, skip them
            if a.box.width < 1 or a.box.length < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = [
            a.box.center_x -0.5*a.box.length,
            a.box.center_y -0.5*a.box.width,
            a.box.center_x +0.5*a.box.length,
            a.box.center_y +0.5*a.box.width,
            ]
            annotation[0, 4] = a.type-1
            annotations = np.append(annotations, annotation, axis=0)


        return annotations

    def num_classes(self):
        return 4

def scale_labels(labels, cam="FRONT"):
    if cam in ["SIDE_LEFT", "SIDE_RIGHT"]:
        x_dim = BIG_X
    else:
        x_dim = SMALL_X
    res = {i + 1: [] for i in range(4)}
    for i in labels:
        res[i.type].append(
            {
                "id": i.id,
                "x": i.box.center_x / x_dim,
                "y": i.box.center_y / Y_DIM,
                "width": i.box.width / x_dim,
                "length": i.box.length / Y_DIM,
            }
        )
    return res


def load_pickle(fpath):
    with open(fpath, "rb") as f:
        return pickle.load(f)


def convert_to_heatmap(img_dims, labels):
    # TODO
    pass
