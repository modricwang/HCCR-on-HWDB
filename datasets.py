import os
import cv2
import numpy as np
import random
import torch

from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from torchvision import datasets, transforms
import torch.utils.data as data


# def my_collate(batch):
#     data = [item[0] for item in batch]
#     target = [item[1] for item in batch]
#     target = torch.LongTensor(target)
#     return [data, target]

def get_train_loader(args):
    dataset = HCCRTrainSet(args)

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.workers,
        # collate_fn=my_collate,
        pin_memory=True)


def get_test_loader(args):
    dataset = HCCRTestSet(args)

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.workers,
        # collate_fn=my_collate,
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

        # self.mean = [0.485, 0.456, 0.406]
        # self.dev = [0.229, 0.224, 0.225]
        #
        # self.transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=self.mean, std=self.dev)])

        self.transform = transforms.ToTensor()

    def __getitem__(self, index):
        image = cv2.imread(self.images[index], cv2.IMREAD_GRAYSCALE)
        # image = cv2.cvtColor(image, cv2.IMREAD_GRAYSCALE)

        max_len = max(image.shape[0], image.shape[1])
        scale_rate = 224.0 / max_len
        image = cv2.resize(image, (int(image.shape[0] * scale_rate), int(image.shape[1] * scale_rate)))
        # print(type(image))

        image2 = np.ndarray(shape=(224, 224), dtype=float)

        # print(image.shape[0])
        delta0 = 224 - image.shape[0]
        delta1 = 224 - image.shape[1]
        image2 = cv2.copyMakeBorder(image, delta0 // 2, delta0 - delta0 // 2, delta1 // 2, delta1 - delta1 // 2,
                                    cv2.BORDER_CONSTANT, value=255)
        # image2 = np.zeros((224, 224))
        # image2.data[112 - image.shape[0] / 2:112 + image.shape[0] / 2,
        # 112 - image.shape[1] / 2:112 + image.shape[1] / 2] = image

        # cv2.resize(image, image2, )
        cv2.threshold(image2, 224, 255, cv2.THRESH_BINARY, image2)
        # cv2.imwrite('data/sample/%d.jpg' % (index), image2)
        image2 = np.expand_dims(image2, axis=2)

        # print(image2.shape)

        # image = cv2.resize(image, (224, 224))
        # image = image.transpose((2, 0, 1))

        image2 = self.transform(image2)

        return image2, self.targets[index]

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

        self.mean = [0.485, 0.456, 0.406]
        self.dev = [0.229, 0.224, 0.225]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.dev)])

    def __getitem__(self, index):
        image = cv2.imread(self.images[index], cv2.IMREAD_GRAYSCALE)
        # image = cv2.cvtColor(image, cv2.IMREAD_GRAYSCALE)

        max_len = max(image.shape[0], image.shape[1])
        scale_rate = 224.0 / max_len
        image = cv2.resize(image, (int(image.shape[0] * scale_rate), int(image.shape[1] * scale_rate)))
        # print(type(image))

        image2 = np.ndarray(shape=(224, 224), dtype=float)

        # print(image.shape[0])
        delta0 = 224 - image.shape[0]
        delta1 = 224 - image.shape[1]
        image2 = cv2.copyMakeBorder(image, delta0 // 2, delta0 - delta0 // 2, delta1 // 2, delta1 - delta1 // 2,
                                    cv2.BORDER_CONSTANT, value=255)
        # image2 = np.zeros((224, 224))
        # image2.data[112 - image.shape[0] / 2:112 + image.shape[0] / 2,
        # 112 - image.shape[1] / 2:112 + image.shape[1] / 2] = image

        # cv2.resize(image, image2, )
        cv2.threshold(image2, 224, 255, cv2.THRESH_BINARY, image2)
        # cv2.imwrite('data/sample/%d.jpg' % (index), image2)
        image2 = np.expand_dims(image2, axis=2)
        # print(image2.shape)

        # image = cv2.resize(image, (224, 224))
        # image = image.transpose((2, 0, 1))

        image2 = self.transform(image2)

    def __len__(self):
        return len(self.targets)
