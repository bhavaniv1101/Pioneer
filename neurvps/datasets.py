import os
import json
import math
import random
import os.path as osp
from glob import glob

import numpy as np
import torch
import skimage.io
import numpy.linalg as LA
import matplotlib.pyplot as plt
import skimage.transform
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from neurvps.config import C


class TrialDataset(Dataset):
    """Class representing `trial` dataset."""

    def __init__(self, rootdir, split, num_validation=400):
        self.rootdir = rootdir
        self.split = split
        # FIXME: Check directory (`rootdir` is set to `data/trial` in `config.yaml`)
        filelist = sorted(glob(f"{rootdir}/image_*.jpg"))
        if split == "train":
            self.filelist = filelist[num_validation :]
            # FIXME: What should `augmentation_level` be (currently set to 2 in `config.yaml`)?
            self.size = len(self.filelist) * C.io.augmentation_level
        elif split == "valid":
            self.filelist = filelist[: num_validation]
            self.size = len(self.filelist)
        print(f"n{split}:", self.size)

    def __len__(self):
        return self.size

    # def __getitem__(self, idx):
    #     # Read in specified image as a NumPy array of shape `(R, C, 3)`
    #     image_name = self.filelist[idx % len(self.filelist)]
    #     image = skimage.io.imread(image_name).astype(float)[:, :, : 3]
    #     # Put last axis (with RGB values) in first position (new shape: `(3, R, C)`)
    #     image = np.rollaxis(image, 2).copy()
    #     # Read in vanishing point mask for this image (assuming that image directory itself
    #     # contains masks for all images, i.e., no separate `vanishing_point_masks` directory)
    #     vpoint_mask_name = image_name.replace("image", "mask")
    #     vpoint_mask = skimage.io.imread(f"{vpoint_mask_name}").astype(float)[:, :, : 3]
    #     # R = len(vpoint_mask[0])
    #     # C = len(vpoint_mask[0][0])
    #     skimage.io.imsave(f"{self.rootdir}skimage_{vpoint_mask_name}.jpg", )
    #     with np.load(f"{vpoint_mask_name}") as npz:
    #         vpts = npz["vpts"]
    #     return torch.tensor(image).float(), {"vpts": torch.tensor(vpts).float()}

    def __getitem__(self, idx):
        # Read in specified image as a NumPy array of shape `(R, C, 3)`
        image_name = self.filelist[idx % len(self.filelist)]
        image = skimage.io.imread(image_name).astype(float)[:, :, : 3]
        n_rows_orig, n_cols_orig = image.shape[: 2]
        # Resize and put last axis (with RGB values) in first position 
        # (new shape: `(3, 512, 512)`)
        image = skimage.transform.resize(image, (512, 512))
        image = np.rollaxis(image, 2).copy()
        # Read in vanishing point for this image (3 numbers on a line in a text file)
        vpoint_fname = f"{self.rootdir}/vpoint_{idx}.txt"
        vpoint = np.loadtxt(vpoint_fname)  # shape is (3,)
        # Scale vanishing point since image has been resized to 512 x 512
        # NOTE: By comparison with TMM17 training data, hypothesis is that
        # vanishing point should be represented as (v_x, v_y, 1), where v_x and
        # v_y are both in the range [-1, 1] (so image center is the origin) and
        # x-axis points right and y-axis points up (as usual)
        vpoint[0] = vpoint[0] * 2 / n_cols_orig  # x-coordinate of v-point
        vpoint[1] = -vpoint[1] * 2 / n_rows_orig  # y-coordinate of v-point
        # print(vpoint)
        vpoint /= np.linalg.norm(vpoint)  # Normalize, as expected by NeurVPS
        # # Read in vanishing point mask for this image (assuming that image directory itself
        # # contains masks for all images, i.e., no separate `vanishing_point_masks` directory)
        return torch.tensor(image).float(), {"vpts": torch.tensor(vpoint[np.newaxis, :]).float()}


class WireframeDataset(Dataset):
    def __init__(self, rootdir, split):
        self.rootdir = rootdir
        filelist = sorted(glob(f"{rootdir}/*/*.png"))

        self.split = split
        if split == "train":
            self.filelist = filelist[500:]
            self.size = len(self.filelist) * C.io.augmentation_level
        elif split == "valid":
            self.filelist = [f for f in filelist[:500] if "a1" not in f]
            self.size = len(self.filelist)
        print(f"n{split}:", self.size)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        iname = self.filelist[idx % len(self.filelist)]
        image = skimage.io.imread(iname).astype(float)[:, :, :3]
        image = np.rollaxis(image, 2).copy()
        with np.load(iname.replace(".png", "_label.npz")) as npz:
            vpts = npz["vpts"]
        return (torch.tensor(image).float(), {"vpts": torch.tensor(vpts).float()})


class ScanNetDataset(Dataset):
    def __init__(self, rootdir, split):
        self.rootdir = rootdir
        self.split = split

        dirs = np.genfromtxt(f"{rootdir}/scannetv2_{split}.txt", dtype=str)
        self.filelist = sum([glob(f"{rootdir}/{d}/*.png") for d in dirs], [])
        if split == "train":
            self.size = len(self.filelist) * C.io.augmentation_level
        elif split == "valid":
            random.seed(0)
            random.shuffle(self.filelist)
            self.filelist = self.filelist[:500]
            self.size = len(self.filelist)
        print(f"n{split}:", self.size)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        iname = self.filelist[idx % len(self.filelist)]
        image = skimage.io.imread(iname)[:, :, :3]
        with np.load(iname.replace("color.png", "vanish.npz")) as npz:
            vpts = np.array([npz[d] for d in ["x", "y", "z"]])
        vpts[:, 1] *= -1
        # plt.imshow(image)
        # cc = ["blue", "cyan", "orange"]
        # for c, w in zip(cc, vpts):
        #     x = w[0] / w[2] * C.io.focal_length * 256 + 256
        #     y = -w[1] / w[2] * C.io.focal_length * 256 + 256
        #     plt.scatter(x, y, color=c)
        #     for xy in np.linspace(0, 512, 10):
        #         plt.plot(
        #             [x, xy, x, xy, x, 0, x, 511],
        #             [y, 0, y, 511, y, xy, y, xy],
        #             color=c,
        #         )
        # plt.show()
        image = np.rollaxis(image.astype(np.float), 2).copy()
        return (torch.tensor(image).float(), {"vpts": torch.tensor(vpts).float()})


class Tmm17Dataset(Dataset):
    def __init__(self, rootdir, split):
        self.rootdir = rootdir
        self.split = split

        filelist = np.genfromtxt(f"{rootdir}/{split}.txt", dtype=str)
        self.filelist = [osp.join(rootdir, f) for f in filelist]
        if split == "train":
            self.size = len(self.filelist) * C.io.augmentation_level
        elif split == "valid":
            self.size = len(self.filelist)
        print(f"n{split}:", self.size)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        iname = self.filelist[idx % len(self.filelist)]
        image = skimage.io.imread(iname)
        tname = iname.replace(".jpg", ".txt")
        axy, bxy = np.genfromtxt(tname, skip_header=1)

        a0, a1 = np.array(axy[:2]), np.array(axy[2:])
        b0, b1 = np.array(bxy[:2]), np.array(bxy[2:])
        xy = intersect(a0, a1, b0, b1) - 0.5
        xy[0] *= 512 / image.shape[1]
        xy[1] *= 512 / image.shape[0]
        image = skimage.transform.resize(image, (512, 512))
        if image.ndim == 2:
            image = image[:, :, None].repeat(3, 2)
        if self.split == "train":
            i, j, h, w = crop(image.shape)
        else:
            i, j, h, w = 0, 0, image.shape[0], image.shape[1]
        image = skimage.transform.resize(image[j : j + h, i : i + w], (512, 512))
        xy[1] = (xy[1] - j) / h * 512
        xy[0] = (xy[0] - i) / w * 512
        # plt.imshow(image)
        # plt.scatter(xy[0], xy[1])
        # plt.show()
        vpts = np.array([[xy[0] / 256 - 1, 1 - xy[1] / 256, C.io.focal_length]])
        vpts[0] /= LA.norm(vpts[0])

        image, vpts = augment(image, vpts, idx // len(self.filelist))
        image = np.rollaxis(image, 2)
        return (torch.tensor(image * 255).float(), {"vpts": torch.tensor(vpts).float()})


def augment(image, vpts, division):
    if division == 1:  # left-right flip
        return image[:, ::-1].copy(), (vpts * [-1, 1, 1]).copy()
    elif division == 2:  # up-down flip
        return image[::-1, :].copy(), (vpts * [1, -1, 1]).copy()
    elif division == 3:  # all flip
        return image[::-1, ::-1].copy(), (vpts * [-1, -1, 1]).copy()
    return image, vpts


def intersect(a0, a1, b0, b1):
    c0 = ccw(a0, a1, b0)
    c1 = ccw(a0, a1, b1)
    d0 = ccw(b0, b1, a0)
    d1 = ccw(b0, b1, a1)
    if abs(d1 - d0) > abs(c1 - c0):
        return (a0 * d1 - a1 * d0) / (d1 - d0)
    else:
        return (b0 * c1 - b1 * c0) / (c1 - c0)


def ccw(c, a, b):
    a0 = a - c
    b0 = b - c
    return a0[0] * b0[1] - b0[0] * a0[1]


def crop(shape, scale=(0.35, 1.0), ratio=(9 / 16, 16 / 9)):
    for attempt in range(20):
        area = shape[0] * shape[1]
        target_area = random.uniform(*scale) * area
        aspect_ratio = random.uniform(*ratio)

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if random.random() < 0.5:
            w, h = h, w

        if h <= shape[0] and w <= shape[1]:
            j = random.randint(0, shape[0] - h)
            i = random.randint(0, shape[1] - w)
            return i, j, h, w

    # Fallback
    w = min(shape[0], shape[1])
    i = (shape[1] - w) // 2
    j = (shape[0] - w) // 2
    return i, j, w, w
