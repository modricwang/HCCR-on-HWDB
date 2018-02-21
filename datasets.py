import os
import cv2
import numpy
import random
import torch

from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from torchvision import datasets, transforms
import torch.utils.data as data


def get_train_loader(args):
    dataset = HCCRTrainSet(args)

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.workers,
        pin_memory=True)


def get_test_loader(args):
    dataset = HCCRTestSet(args)

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.workers,
        pin_memory=True)


class HCCRTrainSet(data.Dataset):
    def __init__(self, args):
        self.images = list()
        self.targets = list()

        for path, _, image_set in os.walk(os.path.join(args.data_dir, 'train')):
            if os.path.isdir(path):
                for image in image_set:
                    self.images.append(os.path.join(path, image))
                    self.targets.append(int(path.split('\\')[-1]))

    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (224, 224))
        image = image.transpose((2, 0, 1))

        return (image, self.targets[index])

    def __len__(self):
        return len(self.targets)


class HCCRTestSet(data.Dataset):
    def __init__(self, args):
        self.images = list()
        self.targets = list()

        for path, _, image_set in os.walk(os.path.join(args.data_dir, 'test')):
            if os.path.isdir(path):
                for image in image_set:
                    self.images.append(os.path.join(path, image))
                    self.targets.append(int(path.split('\\')[-1]))

    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (224, 224))

        image = image.transpose((2, 0, 1))

        return (image, self.targets[index])

    def __len__(self):
        return len(self.targets)
