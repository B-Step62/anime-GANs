import os
import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import core.utils.transforms as tf


class FaceDataset(Dataset):

    def __init__(self, root_paths, imsize, istrain=True):
        super(FaceDataset, self).__init__()
        self.image_paths = sorted(glob.glob(os.path.join(root_paths, '*.png')))
        self.imsize = imsize

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = None
        while image is None:
            image = cv2.imread(self.image_paths[idx])
            idx = np.random.randint(len(self.image_paths))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)

        w, h, _ = image.shape

        # flip
        image = tf.random_flip(image)

        # rotation
        degree = np.random.randint(-30, 30)
        image = tf.rotation(image, degree)

        # crop
        csize = int(w * 0.9) if w < h else int(h * 0.9)
        image = tf.random_crop(image, (csize, csize))

        # resize
        image = tf.rescale(image, (self.imsize, self.imsize))

        # normalize
        image = image - 255 * 0.5
        image = image / (255 * 0.5)
  
        image = torch.from_numpy(image).permute(2,0,1).contiguous().float()
        return image
