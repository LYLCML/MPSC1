# -*- coding: utf-8 -*-

from torchvision.datasets import DatasetFolder
from PIL import Image
from tqdm import tqdm
import itertools
import random


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class IndexedImageFolder(DatasetFolder):
    def __init__(self, roots, train=True, use_cache=False, transform=None, target_transform=None,
                 loader=pil_loader, is_valid_file=None):
        assert isinstance(roots, list), "roots must be a list of dataset paths"

        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform
        self.roots = roots
        self.samples = []
        # for root in roots:
        #     dataset = DatasetFolder(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
        #                      transform=transform,
        #                      target_transform=target_transform,
        #                      is_valid_file=is_valid_file)
        #     self.samples += dataset.samples
        # Load samples from the first dataset, keeping original labels
        first_root = roots[0]
        dataset1 = DatasetFolder(first_root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                 transform=transform,
                                 target_transform=target_transform,
                                 is_valid_file=is_valid_file)
        self.samples += dataset1.samples  # Keep the original labels for the first dataset
        targets = [label for path, label in self.samples]
        num_class= max(targets)
        labels = list(range(num_class + 1))
        labels = itertools.cycle(labels)

        # For the rest of the datasets, set the label to -1
        for root in roots[1:]:
            dataset = DatasetFolder(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                    transform=transform,
                                    target_transform=target_transform,
                                    is_valid_file=is_valid_file)
            if train:
                noise_labels = [next(labels) for _ in range(len(dataset.samples))]
                random.seed(0)
                random.shuffle(noise_labels)
                count = 0
                modified_samples = []
                for path, label in dataset.samples:
                    modified_samples.append([path, noise_labels[count]])
                    count += 1

            else:
                # Modify the label to -1 for all samples in the rest of the datasets
                modified_samples = [(path, -1) for path, label in dataset.samples]
            self.samples += modified_samples  # Append the modified samples

        self.imgs = self.samples  # list, element is (path, label)

        self.use_cache = use_cache
        if self.use_cache:
            self.loaded_samples = self._cache_dataset()  # list, element is (PIL image, label)
        else:
            self.loaded_samples = None

    def _cache_dataset(self):
        cached_dataset = []
        n_samples = len(self.samples)
        print('caching samples ... ')
        for idx, sample in enumerate(tqdm(self.samples, ncols=100, ascii=' >')):
            path, target = sample
            image = self.loader(path)
            item = (image, target)
            cached_dataset.append(item)
        assert len(cached_dataset) == n_samples
        return cached_dataset

    def __getitem__(self, index):
        if self.use_cache:
            assert len(self.loaded_samples) == len(self.samples)
            sample, target = self.loaded_samples[index]
        else:
            path, target = self.samples[index]
            sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return {'index': index, 'data': sample, 'label': target}

