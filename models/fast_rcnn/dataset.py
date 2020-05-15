import os
import torch
import numpy as np
import pickle
import time
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import cv2
import tensorflow.compat.v1 as tf


CAMERAS = ["FRONT", "FRONT_LEFT", "SIDE_LEFT", "FRONT_RIGHT", "SIDE_RIGHT"]


class WaymoDataset(Dataset):
    """Dataset for image segmentation and regression."""

    def __init__(
        self,
        scope="training",
        root_dir="/home/project_x/data/",
        cameras=CAMERAS,
        transform=None,
        mod='fast_rcnn',
    ):

        self.transform = transform
        self.mod = mod
        self.filepaths = []
        root_path = root_dir + scope
        
        if self.mod == 'fast_rcnn':
            for cam in cameras:
                print(root_path + f'/{cam}_with_objects.pickle')
                self.filepaths += load_pickle(root_path + f'/{cam}_with_objects.pickle')
                

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, item):

        fpath = self.filepaths[item]
        raw_sample = load_pickle(fpath)
        img = self.load_image(raw_sample['image'])
        annot = self.load_annotations(raw_sample['labels'].labels)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)
#         sample = {'raw':raw_sample, 'id':fpath.split('/')[-1], **sample}
        sample = {'raw':raw_sample, 'id':fpath, **sample}

        img = torch.as_tensor(sample['img'].float()).permute(2, 0, 1)
        boxes = torch.as_tensor(sample['annot'][:,:4], dtype=torch.float32)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = torch.as_tensor(annot[:,-1], dtype=torch.int64)
        target['area'] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target['image_id'] = sample['id']
        target['scale'] = torch.as_tensor(sample['scale'], dtype=torch.float32)
        
        return img, target

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


def collate_fn(batch):
    return tuple(zip(*batch))
    
def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]

    imgs = torch.from_numpy(np.stack(imgs, axis=0))

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    imgs = imgs.permute(0, 3, 1, 2)

    return {'img': imgs, 'annot': annot_padded} #, 'scale': scales}


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __call__(self, sample, common_size=512):
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape
        if height > width:
            scale = common_size / height
            resized_height = common_size
            resized_width = int(width * scale)
        else:
            scale = common_size / width
            resized_height = int(height * scale)
            resized_width = common_size
        
        image = cv2.resize(image, (resized_width, resized_height))
        new_image = np.zeros((common_size, common_size, 3))
        new_image[0:resized_height, 0:resized_width] = image
        annots[:, :4] *= scale
        return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'scale': scale}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots}

        return sample


class Normalizer(object):

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}

def load_pickle(fpath):
    with open(fpath, "rb") as f:
        return pickle.load(f)