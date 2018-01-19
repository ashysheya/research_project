import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from scipy import misc
import numpy as np
import glob


class SegmentationDataset(Dataset):
    def __init__(self, path_to_images, path_to_segm, transform=None):
        self._images = sorted([file for file in glob.glob(path_to_images + "*.tif")])
        self._segm = sorted([file for file in glob.glob(path_to_segm + '*.tif')])
        self._segm_idx = sorted([int(file[-7:-4]) for file in self._segm])
        self._transform = transform

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        image = misc.imread(self._images[idx])

        if idx in self._segm_idx:
            segm = misc.imread(self._segm[self._segm_idx.index(idx)]).astype(np.int64)
            sample = {'image': image, 'segm': segm, 'has_segm': True, 'idx': idx}
        else:
            segm = np.zeros(image.shape).astype(np.int64)
            sample = {'image': image, 'segm': segm, 'has_segm': False, 'idx': idx}

        if self._transform:
            sample = self._transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self):
        self._totensor = transforms.ToTensor()

    def __call__(self, sample):
        image, segm = sample['image'], sample['segm']

        image = np.expand_dims(image, axis=2)
        return {'image': self._totensor(image),
                'segm': torch.from_numpy(segm),
                'has_segm': sample['has_segm'],
                'idx': sample['idx']}


class Normalize(object):
    def __init__(self):
        self._norm = transforms.Normalize(mean=(0.5,), std=(0.5,))

    def __call__(self, sample):
        return {'image': self._norm(sample['image']),
                'segm': sample['segm'],
                'has_segm': sample['has_segm'],
                'idx': sample['idx']}
